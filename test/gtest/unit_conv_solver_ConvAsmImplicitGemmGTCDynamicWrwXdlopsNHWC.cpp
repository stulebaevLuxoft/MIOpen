/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "unit_conv_solver.hpp"

namespace {

auto GetConvSmokeTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{1, 8, 8, 8}, {8, 8, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetSmokeTestParams(miopenDataType_t datatype)
{
    Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
    if(datatype != miopenBFloat16)
    {
        supportedDevices = supportedDevices | Gpu::gfx908;
    }
    auto testParams = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    testParams.Tunable(5);
    testParams.CheckXnackDisabled();
    testParams.SetTolerance(Gpu::gfx90A | Gpu::gfx950, miopenHalf, 2.0f);

    return testParams;
}

auto GetConvFullTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        // TODO LWPMIOPEN-1314 convert / unify existing ASM solver tests
        // Regression tests for SWDEV-502833
        TestCase{{16, 5, 225, 225}, {64, 5, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{16, 576, 1, 1}, {576, 576, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{16, 2048, 8, 32}, {4096, 2048, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{16, 2048, 8, 32}, {2048, 2048, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{16, 2048, 48, 32}, {2048, 2048, 3, 1}, {0, 0}, {3, 1}, {1, 1}, datatype},
        TestCase{{16, 2048, 1, 512}, {2048, 2048, 1, 2}, {0, 0}, {1, 2}, {1, 1}, datatype},
        TestCase{{16, 2048, 1, 30}, {576, 2048, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetFullTestParams(miopenDataType_t datatype)
{
    Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
    if(datatype != miopenBFloat16)
    {
        supportedDevices = supportedDevices | Gpu::gfx908;
    }
    auto testParams = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
    testParams.Tunable(1000);
    testParams.CheckXnackDisabled();
    testParams.SetTolerance(Gpu::gfx908 | Gpu::gfx94X | Gpu::gfx950, miopenFloat, 3.0f);
    testParams.SetTolerance(Gpu::gfx90A, miopenFloat, 4.0f);
    testParams.SetTolerance(Gpu::gfx94X, miopenFloat, 3.0f);
    testParams.SetTolerance(Gpu::gfx90A, miopenHalf, 2.0f);

    return testParams;
}

} // namespace

using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP16 =
    GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_BFP16 =
    GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP32 =
    GPU_UnitTestConvSolverWrw_FP32;
using CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCDevApplicabilityWrw_NONE =
    CPU_UnitTestConvSolverDevApplicabilityWrw_NONE;

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP16,
       ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC{});
};

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_BFP16,
       ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC{});
};

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP32,
       ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC{});
};

TEST_P(CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCDevApplicabilityWrw_NONE,
       ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP16,
                         testing::Combine(testing::Values(GetSmokeTestParams(miopenHalf)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_BFP16,
    testing::Combine(testing::Values(GetSmokeTestParams(miopenBFloat16)),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetConvSmokeTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP32,
                         testing::Combine(testing::Values(GetSmokeTestParams(miopenFloat)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvSmokeTestCases(miopenFloat))));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP16,
                         testing::Combine(testing::Values(GetFullTestParams(miopenHalf)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_BFP16,
                         testing::Combine(testing::Values(GetFullTestParams(miopenBFloat16)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCWrw_FP32,
                         testing::Combine(testing::Values(GetFullTestParams(miopenFloat)),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvFullTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicWrwXdlopsNHWCDevApplicabilityWrw_NONE,
    testing::Combine(testing::Values(GetSmokeTestParams(miopenFloat)),
                     testing::Values(GetConvSmokeTestCases(miopenFloat)[0])));
