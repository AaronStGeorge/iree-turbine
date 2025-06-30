# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple, no_type_check
import torch

from ..support.ir_imports import (
    RankedTensorType,
    IrType,
)

from ..runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
    impl_helper,
)

from ..support.logging import aot_logger as logger

_templates = impl_helper.JinjaTemplateLoader(__name__)


@CustomOp.register()
class test_add(CustomOp):
    signature = "test_add(Tensor t1, Tensor t2) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_tensor(1)
        t2_desc.specialize_all_dims()
        result_desc = ksel.return_new_tensor(list(t1_desc.t.shape), t1_desc.t.dtype)
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        result_type = kb.arg_bindings[0].type  # type: ignore
        rtt = RankedTensorType(result_type)
        function_name = f"turbine_test_add_jinja_{rtt.rank}d_{str(rtt.element_type)}"
        func_op = _templates.inline_template_function(
            kb,
            "test_add_jinja",
            function_name,
            rank=rtt.rank,
            element_type=str(rtt.element_type),
            tensor_type=str(rtt),
        )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))


@CustomOp.register()
class mm(CustomOp):
    signature = "mm(Tensor t1, Tensor t2) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_tensor(1)
        t2_desc.specialize_all_dims()
        result_desc = ksel.return_new_tensor(
            [t1_desc.t.shape[0], t2_desc.t.shape[1]], t1_desc.t.dtype
        )
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        # Args + result types
        t1_desc = ksel.arg_descs[0]
        t2_desc = ksel.arg_descs[1]
        res_desc = ksel.result_descs[0]

        # Create MLIR type for result
        result_type = IrType.parse(res_desc.mlir_type_asm)

        # Instantiate template
        function_name = (
            f"turbine_mm_{result_type.rank}d_{str(result_type.element_type)}"
        )
        func_op = _templates.inline_template_function(
            kb,
            "mm",
            function_name,
            rank=result_type.rank,
            element_type=result_type.element_type,
            tensor_A_type=t1_desc.mlir_type_asm,
            tensor_B_type=t2_desc.mlir_type_asm,
            tensor_Result_type=res_desc.mlir_type_asm,
        )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))
