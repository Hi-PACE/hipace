#ifndef ROCFFTUTILS_H_
#define ROCFFTUTILS_H_

#include <rocfft.h>

#include <string>


namespace RocFFTUtils
{
    void assert_rocfft_status (std::string const& name, rocfft_status status);

    /** \brief This method converts a cufftResult
     * into the corresponding string
     *
     * @param[in] err a cufftResult
     * @return an std::string
     */
    std::string rocfftErrorToString (const rocfft_status err);
}

#endif // ROCFFTUTILS_H_
