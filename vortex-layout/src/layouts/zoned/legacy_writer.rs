//! Write-time assembly for legacy `vortex.stats` layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::PRUNING_STATS;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::file_stats::StatsAccumulator;
use crate::layouts::zoned::LegacyStatsLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Configuration for writing the legacy `vortex.stats` zoned layout.
pub struct LegacyStatsLayoutOptions {
    /// The number of rows represented by each zone.
    pub block_size: NonZeroUsize,
    /// The statistics to collect for each zone.
    pub stats: Arc<[Stat]>,
    /// Maximum byte length of variable-length min/max statistics.
    pub max_variable_length_statistics_size: usize,
    /// Number of zones whose statistics may be computed concurrently.
    pub concurrency: NonZeroUsize,
}

impl Default for LegacyStatsLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: NonZeroUsize::new(8192)
                .vortex_expect("default block size must be non-zero"),
            stats: PRUNING_STATS.into(),
            max_variable_length_statistics_size: 64,
            concurrency: NonZeroUsize::new(get_available_parallelism().unwrap_or(1))
                .vortex_expect("available parallelism must be non-zero"),
        }
    }
}

/// A layout strategy that emits the release-10-compatible `vortex.stats` wire format.
pub struct LegacyStatsStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    options: LegacyStatsLayoutOptions,
}

impl LegacyStatsStrategy {
    /// Create a legacy stats writer with separate strategies for data and zone-map children.
    pub fn new<Child: LayoutStrategy, Stats: LayoutStrategy>(
        child: Child,
        stats: Stats,
        options: LegacyStatsLayoutOptions,
    ) -> Self {
        Self {
            child: Arc::new(child),
            stats: Arc::new(stats),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for LegacyStatsStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let requested_stats = Arc::clone(&self.options.stats);
        let compute_session = session.clone();
        let accumulator = Arc::new(Mutex::new(StatsAccumulator::new(
            stream.dtype(),
            &requested_stats,
            self.options.max_variable_length_statistics_size,
        )));

        let stream_dtype = stream.dtype().clone();
        let concurrency = self.options.concurrency.get();
        let stream = stream
            .map(move |item| {
                let requested_stats = Arc::clone(&requested_stats);
                let session = compute_session.clone();
                session.handle().spawn_cpu(move || {
                    let (sequence_id, chunk) = item?;
                    chunk
                        .statistics()
                        .compute_all(&requested_stats, &mut session.create_execution_ctx())?;
                    Ok::<_, VortexError>((sequence_id, chunk))
                })
            })
            .buffered(concurrency);

        // The accumulator must observe the computed chunks in stream order so its rows remain
        // aligned with the data zones.
        let ordered_accumulator = Arc::clone(&accumulator);
        let stream = SequentialStreamAdapter::new(
            stream_dtype,
            stream.map(move |item| {
                let (sequence_id, chunk) = item?;
                ordered_accumulator
                    .lock()
                    .push_chunk_without_compute(&chunk)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        // The data child must precede its auxiliary statistics child in sequence order.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        let mut exec_ctx = session.create_execution_ctx();
        let Some((stats_array, present_stats)) = accumulator.lock().as_array(&mut exec_ctx)? else {
            return Ok(data_layout);
        };

        let stats_stream = stats_array
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .stats
            .write_stream(ctx, Arc::clone(&segment_sink), stats_stream, eof, session)
            .await?;

        Ok(LegacyStatsLayout::try_new(
            data_layout,
            zones_layout,
            self.options.block_size,
            present_stats,
        )?
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes() + self.stats.buffered_bytes()
    }
}
