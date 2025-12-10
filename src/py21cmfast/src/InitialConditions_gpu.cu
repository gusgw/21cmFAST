/*
    InitialConditions_gpu.cu -- GPU-accelerated initial conditions computation

    This file implements GPU kernels for computing initial conditions.
    Following the pattern established in filtering.cu:
    - FFT operations remain on CPU (FFTW)
    - Computational kernels run on GPU
    - Data is transferred between CPU and GPU as needed

    Phase 1 Implementation:
    - CPU generates random numbers (GSL) for bit-for-bit reproducibility
    - GPU handles: velocity field computation, complex conjugate adjustment, subsampling
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

// GPU headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "indexing.h"
#include "dft.h"
#include "filtering.h"
#include "cosmology.h"
#include "rng.h"

#include "InitialConditions_gpu.h"

// ============================================================================
// GPU Kernel: Velocity field computation in k-space
// ============================================================================
// Computes velocity component from density field: v_k = i * k_component / k^2 * delta_k
// This is the main parallelizable operation in InitialConditions

__global__ void compute_velocity_kernel(
    cuFloatComplex *box,      // Input/output: k-space field
    int dimension,            // DIM
    int midpoint,             // MIDDLE
    int midpoint_para,        // MIDDLE_PARA
    float delta_k,            // DELTA_K
    float delta_k_para,       // DELTA_K_PARA
    float volume,             // VOLUME
    int component             // 0=x, 1=y, 2=z
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * (midpoint_para + 1);

    if (idx >= num_pixels) return;

    // Compute 3D indices from flattened index
    // C_INDEX(n_x, n_y, n_z) = z + (midpoint_para + 1) * (y + dimension * x)
    int n_z = idx % (midpoint_para + 1);
    unsigned long long remaining = idx / (midpoint_para + 1);
    int n_y = remaining % dimension;
    int n_x = remaining / dimension;

    // Compute wave vector components
    float k_x, k_y, k_z;

    if (n_x > midpoint)
        k_x = (n_x - dimension) * delta_k;
    else
        k_x = n_x * delta_k;

    if (n_y > midpoint)
        k_y = (n_y - dimension) * delta_k;
    else
        k_y = n_y * delta_k;

    k_z = n_z * delta_k_para;

    float k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

    // Handle DC mode
    if (n_x == 0 && n_y == 0 && n_z == 0) {
        box[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // Get the k component for this velocity direction
    float k_comp;
    if (component == 0) k_comp = k_x;
    else if (component == 1) k_comp = k_y;
    else k_comp = k_z;

    // Multiply by i * k_component / k^2 / VOLUME
    // i * (a + bi) = -b + ai
    cuFloatComplex val = box[idx];
    float factor = k_comp / k_sq / volume;
    // Multiply by i: (a + bi) * i = -b + ai
    box[idx] = make_cuFloatComplex(-cuCimagf(val) * factor, cuCrealf(val) * factor);
}

// ============================================================================
// GPU Kernel: Complex conjugate adjustment for Hermitian symmetry
// ============================================================================
// Enforces the complex conjugate relations required for a real-valued FFT result
// This operates on the k=0 and k=N/2 planes

__global__ void adjust_complex_conj_kernel(
    cuFloatComplex *box,
    int dimension,
    int midpoint,
    int midpoint_para
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // This kernel handles the edge cases for Hermitian symmetry
    // We'll launch this with enough threads to cover MIDDLE iterations

    if (idx >= (unsigned long long)midpoint) return;

    int i = idx + 1;  // i goes from 1 to MIDDLE-1

    // j corners (j = 0 or MIDDLE)
    for (int j = 0; j <= midpoint; j += midpoint) {
        for (int k = 0; k <= midpoint_para; k += midpoint_para) {
            unsigned long long src_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * (dimension - i));
            unsigned long long dst_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * i);
            cuFloatComplex src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));
        }
    }

    // All of j (j from 1 to MIDDLE-1)
    for (int j = 1; j < midpoint; j++) {
        for (int k = 0; k <= midpoint_para; k += midpoint_para) {
            // HIRES_box[C_INDEX(i, j, k)] = conjf(HIRES_box[C_INDEX(DIM - i, DIM - j, k)])
            unsigned long long src_idx = k + (midpoint_para + 1) * ((dimension - j) + (unsigned long long)dimension * (dimension - i));
            unsigned long long dst_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * i);
            cuFloatComplex src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));

            // HIRES_box[C_INDEX(i, DIM - j, k)] = conjf(HIRES_box[C_INDEX(DIM - i, j, k)])
            src_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * (dimension - i));
            dst_idx = k + (midpoint_para + 1) * ((dimension - j) + (unsigned long long)dimension * i);
            src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));
        }
    }
}

__global__ void adjust_complex_conj_corners_kernel(
    cuFloatComplex *box,
    int dimension,
    int midpoint,
    int midpoint_para
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle i corners (i = 0 or MIDDLE)
    // j from 1 to MIDDLE-1

    if (idx >= (unsigned long long)(midpoint - 1)) return;

    int j = idx + 1;  // j goes from 1 to MIDDLE-1

    for (int i = 0; i <= midpoint; i += midpoint) {
        for (int k = 0; k <= midpoint_para; k += midpoint_para) {
            // HIRES_box[C_INDEX(i, j, k)] = conjf(HIRES_box[C_INDEX(i, DIM - j, k)])
            unsigned long long src_idx = k + (midpoint_para + 1) * ((dimension - j) + (unsigned long long)dimension * i);
            unsigned long long dst_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * i);
            cuFloatComplex src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));
        }
    }
}

// ============================================================================
// GPU Kernel: Subsample high-res box to low-res
// ============================================================================

__global__ void subsample_box_kernel(
    float *hires_box,         // Input: high-res real-space box (with FFT padding)
    float *lowres_box,        // Output: low-res box (no padding)
    int hii_dim,              // HII_DIM
    int hii_d_para,           // HII_D_PARA
    int dim,                  // DIM
    int mid_para,             // MID_PARA
    float f_pixel_factor,     // DIM / HII_DIM
    float volume              // VOLUME for normalization
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)hii_dim * hii_dim * hii_d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices for low-res box
    int k = idx % hii_d_para;
    unsigned long long remaining = idx / hii_d_para;
    int j = remaining % hii_dim;
    int i = remaining / hii_dim;

    // Map to high-res indices
    int hi = (int)(i * f_pixel_factor + 0.5f);
    int hj = (int)(j * f_pixel_factor + 0.5f);
    int hk = (int)(k * f_pixel_factor + 0.5f);

    // R_FFT_INDEX(x, y, z) = z + 2 * (mid_para + 1) * (y + dim * x)
    unsigned long long hires_idx = hk + 2llu * (mid_para + 1) * (hj + (unsigned long long)dim * hi);

    lowres_box[idx] = hires_box[hires_idx] / volume;
}

// ============================================================================
// GPU Kernel: Copy hires density to output (with normalization)
// ============================================================================

__global__ void copy_hires_density_kernel(
    float *hires_box,         // Input: FFT result (with padding)
    float *output,            // Output: hires_density array (no padding)
    int dim,                  // DIM
    int d_para,               // D_PARA
    int mid_para,             // MID_PARA
    float volume              // VOLUME for normalization
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dim * dim * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dim;
    int i = remaining / dim;

    // R_FFT_INDEX(x, y, z) = z + 2 * (mid_para + 1) * (y + dim * x)
    unsigned long long fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dim * i);

    output[idx] = hires_box[fft_idx] / volume;
}

// ============================================================================
// GPU Kernel: Store velocity to output array
// ============================================================================

__global__ void store_velocity_kernel(
    float *hires_box,         // Input: FFT result (with padding)
    float *output,            // Output: velocity array
    int dimension,            // DIM or HII_DIM
    int d_para,               // D_PARA or HII_D_PARA
    int dim,                  // DIM (for FFT indexing)
    int mid_para,             // MID_PARA
    float f_pixel_factor,     // pixel factor (1.0 for hires, DIM/HII_DIM for lowres)
    bool is_hires             // true for hires, false for lowres
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices for output
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dimension;
    int i = remaining / dimension;

    unsigned long long fft_idx;
    if (is_hires) {
        // Direct copy
        fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dim * i);
    } else {
        // Subsample
        int hi = (int)(i * f_pixel_factor + 0.5f);
        int hj = (int)(j * f_pixel_factor + 0.5f);
        int hk = (int)(k * f_pixel_factor + 0.5f);
        fft_idx = hk + 2llu * (mid_para + 1) * (hj + (unsigned long long)dim * hi);
    }

    output[idx] = hires_box[fft_idx];
}

// ============================================================================
// Main GPU function: ComputeInitialConditions_gpu
// ============================================================================

extern "C" int ComputeInitialConditions_gpu(unsigned long long random_seed, InitialConditions *boxes) {
    int status;
    cudaError_t err;

    Try {
        LOG_DEBUG("ComputeInitialConditions_gpu: Starting GPU-accelerated computation");

#if LOG_LEVEL >= DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeMatterOptions(matter_options_global);
        writeCosmoParams(cosmo_params_global);
#endif

        int n_x, n_y, n_z, i, j, k, ii;
        float k_x, k_y, k_z, k_mag, p, a, b;
        float f_pixel_factor;
        int dimension;

        // Initialize RNG (CPU - for bit-for-bit reproducibility with CPU version)
        gsl_rng *r[simulation_options_global->N_THREADS];
        seed_rng_threads(r, random_seed);
        omp_set_num_threads(simulation_options_global->N_THREADS);

        dimension = matter_options_global->PERTURB_ON_HIGH_RES ? simulation_options_global->DIM
                                                               : simulation_options_global->HII_DIM;

        // Allocate CPU arrays
        fftwf_complex *HIRES_box =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        fftwf_complex *HIRES_box_saved =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        if (!HIRES_box || !HIRES_box_saved) {
            LOG_ERROR("Failed to allocate CPU memory for HIRES boxes");
            Throw(MemoryAllocError);
        }

        f_pixel_factor = simulation_options_global->DIM / (float)simulation_options_global->HII_DIM;

        // ============ CPU: Generate k-space Gaussian random field ============
        // This stays on CPU for bit-for-bit reproducibility with the CPU version
        // Use float pointer to access real/imag parts (CUDA-compatible, avoids C complex.h issues)
        LOG_DEBUG("Generating k-space Gaussian random field on CPU");

        init_ps();

        float *hires_float = (float *)HIRES_box;

#pragma omp parallel shared(hires_float, r) private(n_x, n_y, n_z, k_x, k_y, k_z, k_mag, p, a, b) \
    num_threads(simulation_options_global->N_THREADS)
        {
            int thread_num = omp_get_thread_num();
#pragma omp for
            for (n_x = 0; n_x < simulation_options_global->DIM; n_x++) {
                if (n_x > MIDDLE)
                    k_x = (n_x - simulation_options_global->DIM) * DELTA_K;
                else
                    k_x = n_x * DELTA_K;

                for (n_y = 0; n_y < simulation_options_global->DIM; n_y++) {
                    if (n_y > MIDDLE)
                        k_y = (n_y - simulation_options_global->DIM) * DELTA_K;
                    else
                        k_y = n_y * DELTA_K;

                    for (n_z = 0; n_z <= MIDDLE_PARA; n_z++) {
                        k_z = n_z * DELTA_K_PARA;
                        k_mag = sqrtf(k_x * k_x + k_y * k_y + k_z * k_z);
                        p = power_in_k(k_mag);

                        a = gsl_ran_ugaussian(r[thread_num]);
                        b = gsl_ran_ugaussian(r[thread_num]);

                        float scale = sqrtf(VOLUME * p / 2.0f);
                        unsigned long long idx = C_INDEX(n_x, n_y, n_z);
                        hires_float[2 * idx] = scale * a;      // real part
                        hires_float[2 * idx + 1] = scale * b;  // imag part
                    }
                }
            }
        }
        LOG_DEBUG("Generated random field");

        // ============ CPU: Adjust complex conjugates ============
        // Keep on CPU for now (small operation, complex indexing)
        // Using float pointer to access real/imag parts directly (CUDA-compatible)

        float *box_float = (float *)HIRES_box;

        // Helper macro for accessing real and imag parts
        #define BOX_REAL(idx) box_float[2*(idx)]
        #define BOX_IMAG(idx) box_float[2*(idx) + 1]

        // corners - set to real-only (zero imag) or zero
        BOX_REAL(C_INDEX(0, 0, 0)) = 0; BOX_IMAG(C_INDEX(0, 0, 0)) = 0;

        BOX_IMAG(C_INDEX(0, 0, MIDDLE_PARA)) = 0;
        BOX_IMAG(C_INDEX(0, MIDDLE, 0)) = 0;
        BOX_IMAG(C_INDEX(0, MIDDLE, MIDDLE_PARA)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, 0, 0)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, 0, MIDDLE_PARA)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, MIDDLE, 0)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, MIDDLE, MIDDLE_PARA)) = 0;

#pragma omp parallel shared(box_float) private(i, j, k) num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (i = 1; i < MIDDLE; i++) {
                for (j = 0; j <= MIDDLE; j += MIDDLE) {
                    for (k = 0; k <= MIDDLE_PARA; k += MIDDLE_PARA) {
                        unsigned long long src_idx = C_INDEX((simulation_options_global->DIM) - i, j, k);
                        unsigned long long dst_idx = C_INDEX(i, j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);  // conjugate
                    }
                }
                for (j = 1; j < MIDDLE; j++) {
                    for (k = 0; k <= MIDDLE_PARA; k += MIDDLE_PARA) {
                        unsigned long long src_idx = C_INDEX((simulation_options_global->DIM) - i,
                                                             (simulation_options_global->DIM) - j, k);
                        unsigned long long dst_idx = C_INDEX(i, j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);

                        src_idx = C_INDEX((simulation_options_global->DIM) - i, j, k);
                        dst_idx = C_INDEX(i, (simulation_options_global->DIM) - j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);
                    }
                }
            }
        }

#pragma omp parallel shared(box_float) private(i, j, k) num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (i = 0; i <= MIDDLE; i += MIDDLE) {
                for (j = 1; j < MIDDLE; j++) {
                    for (k = 0; k <= MIDDLE_PARA; k += MIDDLE_PARA) {
                        unsigned long long src_idx = C_INDEX(i, (simulation_options_global->DIM) - j, k);
                        unsigned long long dst_idx = C_INDEX(i, j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);
                    }
                }
            }
        }

        #undef BOX_REAL
        #undef BOX_IMAG

        LOG_DEBUG("Adjusted complex conjugates");

        // Save the k-space field for later use
        memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        LOG_SUPER_DEBUG("Saved k-space field");

        // ============ CPU: FFT to real space ============
        int stat = dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                                D_PARA, simulation_options_global->N_THREADS, HIRES_box);
        if (stat > 0) Throw(stat);
        LOG_DEBUG("FFT to real space complete");

        // ============ GPU: Copy hires density to output ============
        {
            size_t fft_size = TOT_FFT_NUM_PIXELS * sizeof(float);
            size_t out_size = TOT_NUM_PIXELS * sizeof(float);

            float *d_hires_box, *d_output;
            err = cudaMalloc(&d_hires_box, fft_size);
            if (err != cudaSuccess) {
                LOG_ERROR("CUDA malloc failed for d_hires_box: %s", cudaGetErrorString(err));
                Throw(CUDAError);
            }

            err = cudaMalloc(&d_output, out_size);
            if (err != cudaSuccess) {
                cudaFree(d_hires_box);
                LOG_ERROR("CUDA malloc failed for d_output: %s", cudaGetErrorString(err));
                Throw(CUDAError);
            }

            err = cudaMemcpy(d_hires_box, (float *)HIRES_box, fft_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int threadsPerBlock = 256;
            int numBlocks = (TOT_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

            copy_hires_density_kernel<<<numBlocks, threadsPerBlock>>>(
                d_hires_box, d_output,
                simulation_options_global->DIM,
                D_PARA, MID_PARA, VOLUME
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(boxes->hires_density, d_output, out_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            cudaFree(d_hires_box);
            cudaFree(d_output);
        }
        LOG_DEBUG("Saved hires_density");

        // ============ Create low-res density field ============
        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
            filter_box(HIRES_box, 0, 0,
                       L_FACTOR * simulation_options_global->BOX_LEN /
                           (simulation_options_global->HII_DIM + 0.0),
                       0.);
        }

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, HIRES_box);

        // ============ GPU: Subsample to low-res ============
        {
            size_t fft_size = TOT_FFT_NUM_PIXELS * sizeof(float);
            size_t lowres_size = HII_TOT_NUM_PIXELS * sizeof(float);

            float *d_hires_box, *d_lowres_box;
            err = cudaMalloc(&d_hires_box, fft_size);
            CATCH_CUDA_ERROR(err);
            err = cudaMalloc(&d_lowres_box, lowres_size);
            if (err != cudaSuccess) {
                cudaFree(d_hires_box);
                CATCH_CUDA_ERROR(err);
            }

            err = cudaMemcpy(d_hires_box, (float *)HIRES_box, fft_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int threadsPerBlock = 256;
            int numBlocks = (HII_TOT_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

            subsample_box_kernel<<<numBlocks, threadsPerBlock>>>(
                d_hires_box, d_lowres_box,
                simulation_options_global->HII_DIM,
                HII_D_PARA,
                simulation_options_global->DIM,
                MID_PARA,
                f_pixel_factor,
                VOLUME
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(boxes->lowres_density, d_lowres_box, lowres_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            cudaFree(d_hires_box);
            cudaFree(d_lowres_box);
        }
        LOG_DEBUG("Created lowres_density");

        // ============ Velocity fields ============
        // Allocate GPU memory for velocity computation
        size_t kspace_size = KSPACE_NUM_PIXELS * sizeof(fftwf_complex);
        size_t fft_size = TOT_FFT_NUM_PIXELS * sizeof(float);

        cuFloatComplex *d_kspace_box;
        float *d_realspace_box, *d_output_box;

        err = cudaMalloc(&d_kspace_box, kspace_size);
        CATCH_CUDA_ERROR(err);
        err = cudaMalloc(&d_realspace_box, fft_size);
        if (err != cudaSuccess) {
            cudaFree(d_kspace_box);
            CATCH_CUDA_ERROR(err);
        }

        size_t vel_output_size;
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            vel_output_size = TOT_NUM_PIXELS * sizeof(float);
        } else {
            vel_output_size = HII_TOT_NUM_PIXELS * sizeof(float);
        }

        err = cudaMalloc(&d_output_box, vel_output_size);
        if (err != cudaSuccess) {
            cudaFree(d_kspace_box);
            cudaFree(d_realspace_box);
            CATCH_CUDA_ERROR(err);
        }

        for (ii = 0; ii < 3; ii++) {
            memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

            // ============ GPU: Compute velocity in k-space ============
            err = cudaMemcpy(d_kspace_box, HIRES_box, kspace_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int threadsPerBlock = 256;
            int numBlocks = (KSPACE_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

            compute_velocity_kernel<<<numBlocks, threadsPerBlock>>>(
                d_kspace_box,
                simulation_options_global->DIM,
                MIDDLE, MIDDLE_PARA,
                DELTA_K, DELTA_K_PARA,
                VOLUME,
                ii
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(HIRES_box, d_kspace_box, kspace_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            // Filter if needed
            if (!matter_options_global->PERTURB_ON_HIGH_RES) {
                if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                    filter_box(HIRES_box, 0, 0,
                               L_FACTOR * simulation_options_global->BOX_LEN /
                                   (simulation_options_global->HII_DIM + 0.0),
                               0.);
                }
            }

            // CPU: FFT to real space
            dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_box);

            // ============ GPU: Store velocity to output ============
            err = cudaMemcpy(d_realspace_box, (float *)HIRES_box, fft_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int vel_dimension, vel_d_para;
            unsigned long long vel_num_pixels;
            float *output_ptr;

            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                vel_dimension = simulation_options_global->DIM;
                vel_d_para = D_PARA;
                vel_num_pixels = TOT_NUM_PIXELS;
                if (ii == 0) output_ptr = boxes->hires_vx;
                else if (ii == 1) output_ptr = boxes->hires_vy;
                else output_ptr = boxes->hires_vz;
            } else {
                vel_dimension = simulation_options_global->HII_DIM;
                vel_d_para = HII_D_PARA;
                vel_num_pixels = HII_TOT_NUM_PIXELS;
                if (ii == 0) output_ptr = boxes->lowres_vx;
                else if (ii == 1) output_ptr = boxes->lowres_vy;
                else output_ptr = boxes->lowres_vz;
            }

            numBlocks = (vel_num_pixels + threadsPerBlock - 1) / threadsPerBlock;

            store_velocity_kernel<<<numBlocks, threadsPerBlock>>>(
                d_realspace_box, d_output_box,
                vel_dimension, vel_d_para,
                simulation_options_global->DIM, MID_PARA,
                f_pixel_factor,
                matter_options_global->PERTURB_ON_HIGH_RES
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(output_ptr, d_output_box, vel_output_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);
        }

        cudaFree(d_kspace_box);
        cudaFree(d_realspace_box);
        cudaFree(d_output_box);

        LOG_DEBUG("Computed velocity fields");

        // ============ Cleanup ============
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

        fftwf_free(HIRES_box);
        fftwf_free(HIRES_box_saved);

        free_ps();
        free_rng_threads(r);

        LOG_DEBUG("ComputeInitialConditions_gpu: Complete");

    }  // End of Try{}

    Catch(status) { return (status); }
    return (0);
}
