// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Custom layout-reader resolution for SmithDB file scans.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::DataFusionError;
use datafusion_datasource::PartitionedFile;
use vortex::layout::LayoutReader;
use vortex::session::VortexSession;

/// Shared reference to a Vortex layout reader.
pub type LayoutReaderRef = Arc<dyn LayoutReader>;

/// Resolves a layout reader for a DataFusion partitioned file.
///
/// SmithDB uses this hook to route a logical segment to a composite reader
/// backed by auxiliary index files. Implementations own any I/O and cache
/// required to build the reader.
#[async_trait]
pub trait LayoutReaderResolver: Send + Sync + std::fmt::Debug {
    /// Resolves the layout reader for `file`.
    async fn resolve(
        &self,
        session: &VortexSession,
        file: &PartitionedFile,
    ) -> Result<LayoutReaderRef, DataFusionError>;
}
