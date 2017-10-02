#pragma ONCE

#include <chrono>
#include <string>

#include "logging.hpp"

using namespace std::chrono;
using watch = std::chrono::high_resolution_clock;

template <class t_dur = std::chrono::seconds> struct cl_timer {
    watch::time_point start;
    std::string name;
    bool output;
    cl_timer(const std::string& _n, bool o = true)
        : name(_n)
        , output(o)
    {
        if (output)
            LOG_INFO << "START(" << name << ")";
        start = watch::now();
    }
    ~cl_timer()
    {
        auto stop = watch::now();
        auto time_spent = stop - start;
        if (output)
            LOG_INFO << "STOP(" << name << ") - "
                     << duration_cast<t_dur>(time_spent).count() << " sec";
    }
    watch::duration elapsed() const { return watch::now() - start; }
};

template <class t_dur = std::chrono::seconds> struct cl_read_timer {
    watch::time_point start;
    std::string name;
    size_t bytes;
    bool output;
    cl_read_timer(const std::string& _n,size_t rb, bool o = true)
        : name(_n)
	, bytes(rb)
        , output(o)
    {
        if (output)
            LOG_INFO << "START(" << name << ")";
        start = watch::now();
    }
    ~cl_read_timer()
    {
        auto stop = watch::now();
        auto time_spent = stop - start;
        if (output)
            LOG_INFO << "STOP(" << name << ") - "
                     << duration_cast<t_dur>(time_spent).count() << " sec ("
		     << float(bytes/(1024*1024))/float(duration_cast<t_dur>(time_spent).count()) << " MiB/s)"; 
			
    }
    watch::duration elapsed() const { return watch::now() - start; }
};
