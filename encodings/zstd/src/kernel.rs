// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementExecuteAdaptor;
use vortex_session::VortexSession;

use crate::Zstd;

pub(super) fn initialize(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        ListContains.id(),
        Zstd,
        ListContainsElementExecuteAdaptor(Zstd),
    );
}
