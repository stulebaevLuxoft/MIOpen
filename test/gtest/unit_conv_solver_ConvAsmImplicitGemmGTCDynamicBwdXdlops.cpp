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

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{64, 64, 28, 28}, {16, 64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

auto GetConvTestCasesFull(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    auto cases = std::vector{
        // clang-format off
        TestCase{{ 64,   64, 28, 28}, {  16,   64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 16,  128, 36, 36}, {  32,  128, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 64,   64, 56, 56}, { 256,   64, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{ 64,  224, 17, 17}, { 224,  224, 1, 7}, {0, 3}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  128, 35, 35}, { 256,  128, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        TestCase{{128,  128, 64, 64}, { 256,  128, 3, 3}, {1, 1}, {2, 2}, {1, 1}, datatype},
        TestCase{{128,  768, 17, 17}, { 256,  768, 3, 3}, {1, 1}, {1, 1}, {2, 2}, datatype},
        TestCase{{  3,  256, 28, 28}, {  80,  256, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{  2,  256, 12, 18}, { 256,  256, 3, 3}, {1, 1}, {1, 1}, {1, 1}, datatype},
        TestCase{{400,  256,  7,  7}, {1024,  256, 7, 7}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{400,  256,  1,  1}, {1024,  256, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype},
        TestCase{{  8,   16,  5,  5}, {   8,   16, 2, 2}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // ho=wo=1 stride=2
        TestCase{{256, 2048,  2,  2}, {1024, 2048, 1, 1}, {0, 0}, {2, 2}, {1, 1}, datatype},
        // clang-format on
    };

    if(datatype == miopenHalf)
    {
        // clang-format off
        // Group
        cases.emplace_back(TestCase{
            {datatype, {64, 32, 28, 28}},
            {datatype, {16, 16,  1,  1}},
            datatype,
            {{0, 0}, {1, 1}, {1, 1}, 2}
        });
        cases.emplace_back(TestCase{
            {datatype, {16, 64, 17, 17}},
            {datatype, {64, 16,  1,  1}},
            datatype,
            {{0, 0}, {1, 1}, {1, 1}, 4}
        });
        cases.emplace_back(TestCase{
            {datatype, {  8, 128, 56, 56}},
            {datatype, {128,  16,  3,  3}},
            datatype,
            {{1, 1}, {1, 1}, {1, 1}, 8}
        });
        // clang-format on
    }

    if(datatype == miopenFloat)
    {
        // clang-format off
        cases.emplace_back(TestCase{{4, 512, 128, 128}, {12, 512, 1, 1}, {0, 0}, {1, 1}, {1, 1}, datatype});
        // clang-format on
    }

    return cases;
}

const auto& GetTestParams()
{
    static const auto params = [] {
        auto p = miopen::unit_tests::UnitTestConvSolverParams(Gpu::gfx908);
        p.CheckXnackDisabled();
        p.SetTolerance(Gpu::gfx908, miopenFloat, 2.0f);
        return p;
    }();
    return params;
}

} // namespace

using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP16 =
    GPU_UnitTestConvSolverBwd_FP16;
using GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP32 =
    GPU_UnitTestConvSolverBwd_FP32;
using CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsDevApplicabilityBwd_NONE =
    CPU_UnitTestConvSolverDevApplicabilityBwd_NONE;

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP16,
       ConvAsmImplicitGemmGTCDynamicBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlops{});
};

TEST_P(GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP32,
       ConvAsmImplicitGemmGTCDynamicBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlops{});
};

TEST_P(CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsDevApplicabilityBwd_NONE,
       ConvAsmImplicitGemmGTCDynamicBwdXdlops)
{
    this->RunTest(miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlops{});
};

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Device applicability test
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsDevApplicabilityBwd_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetConvTestCases(miopenFloat)[0])));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestConvSolverAsmImplicitGemmGTCDynamicXdlopsBwd_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoImplicitGEMM),
                                          testing::ValuesIn(GetConvTestCasesFull(miopenFloat))));
