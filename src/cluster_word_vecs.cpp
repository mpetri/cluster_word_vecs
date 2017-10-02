#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <experimental/string_view>

#include "kmcuda.h"

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/progress.hpp>

#include "logging.hpp"
#include "timing.hpp"
#include "util.hpp"

namespace po = boost::program_options;

struct vector_data {
    std::vector<float> dat;
    size_t num_samples = 0;
    size_t num_features = 0;
    std::vector<std::string> word_str;
};

struct vector_data_f16 {
    std::vector<uint16_t> dat;
    size_t num_samples = 0;
    size_t num_features = 0;
    std::vector<std::string> word_str;
};


const char*
parse_line(const char* cur_line,char* word_buf,float* float_data,size_t max_word_len,size_t cols)
{
    const char* tmp = cur_line;
    /* extract word token */
    size_t word_len = 0;
    while(*tmp != ' ') {
	word_buf[word_len++] = *tmp++;
    }
    word_buf[word_len] = 0;
    tmp++;
    if(word_len > max_word_len) {
	while(*tmp != '\n') ++tmp;
    } else {
	for(size_t i=0;i<cols;i++) {
        	*float_data++ = (float) fast_atof(tmp);
    	}
    }
    while(*tmp != '\n') tmp++;
    return ++tmp;
}

const char*
parse_line_f16(const char* cur_line,char* word_buf,uint16_t* float_data,size_t max_word_len,size_t cols)
{
    const char* tmp = cur_line;
    /* extract word token */
    size_t word_len = 0;
    while(*tmp != ' ') {
	word_buf[word_len++] = *tmp++;
    }
    word_buf[word_len] = 0;
    tmp++;
    if(word_len > max_word_len) {
	while(*tmp != '\n') ++tmp;
    } else {
	size_t written_floats = 0;
	for(size_t i=0;i<cols;i++) {
		float f = (float) fast_atof(tmp);
		uint32_t f32 = *((uint32_t*)(&f));
		uint16_t f16 = basetable[(f32>>23)&0x1ff]+((f32&0x007fffff)>>shifttable[(f32>>23)&0x1ff]);
        	*float_data++ = f16;
		written_floats++;
    	}
    }
    while(*tmp != '\n') tmp++;
    return ++tmp;
}


vector_data read_vector_data(std::string file_name,size_t max_word_len)
{
    cl_timer<> cluster_start("read_vector_data");
    LOG_INFO << "Loading word vector data from " << file_name;
    vector_data vd;
    FILE* f = fopen(file_name.c_str(), "r");
    // Determine file size
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    char* file_data = new char[size+1];
    rewind(f);
    {
	cl_read_timer<> cluster_start("fread vector data file",size);
    	fread(file_data, sizeof(char), size, f);
    }  
    file_data[size] = 0;
    fclose(f);

    const char* cur_line = file_data;
    int rows;int cols;
    sscanf(cur_line,"%d %d\n",&rows,&cols);
    float* float_data = new float[cols+1];
    char word_buf[1024] = {0};
    while(*cur_line != '\n') cur_line++;
    cur_line++;
    boost::progress_display pd(rows);
    size_t skipped_words = 0;
    vd.dat.reserve(rows*cols);
    std::cout << "rows = " << rows << " cols = " << cols << std::endl;
    auto prev_line = (const char*) file_data;
    while( *cur_line != 0 ) {
	prev_line = cur_line;
	cur_line = parse_line(cur_line,word_buf,float_data,max_word_len,cols);
	if(strlen(word_buf) > max_word_len) {
		skipped_words++;
		++pd;
		continue;
	}
        vd.word_str.emplace_back(word_buf);
	vd.dat.insert(vd.dat.end(), float_data,float_data+cols);
	++pd;
        vd.num_samples++;
    }
    LOG_INFO << "skipped words = " << skipped_words << " (" 
	<< float(skipped_words)/float(rows)*100.0 << "%)";
    vd.num_features = cols;
    delete[] float_data;
    return vd;
}

vector_data_f16 read_vector_data_f16(std::string file_name,size_t max_word_len)
{
    generatetables();
    cl_timer<> cluster_start("read_vector_data");
    LOG_INFO << "Loading word vector data from " << file_name;
    vector_data_f16 vd;
    FILE* f = fopen(file_name.c_str(), "r");
    // Determine file size
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    char* file_data = new char[size+1];
    rewind(f);
    {
	cl_read_timer<> cluster_start("fread vector data file",size);
    	fread(file_data, sizeof(char), size, f);
    }  
    file_data[size] = 0;
    fclose(f);

    const char* cur_line = file_data;
    int rows;int cols;
    sscanf(cur_line,"%d %d\n",&rows,&cols);
    uint16_t* float_data = new uint16_t[cols];
    char word_buf[1024] = {0};
    while(*cur_line != '\n') cur_line++;
    cur_line++;
    boost::progress_display pd(rows);
    size_t skipped_words = 0;
    vd.dat.reserve(rows*cols);
    vd.word_str.reserve(rows);
    std::cout << "rows = " << rows << " cols = " << cols << std::endl;
    auto prev_line = (const char*) file_data;
    while( *cur_line != 0 ) {
	prev_line = cur_line;
	cur_line = parse_line_f16(cur_line,word_buf,float_data,max_word_len,cols);
	if(strlen(word_buf) > max_word_len) {
		skipped_words++;
		++pd;
		continue;
	}
        vd.word_str.emplace_back(word_buf,strlen(word_buf));
	vd.dat.insert(vd.dat.end(), float_data,float_data+cols);
/*
        std::cout << " word = '" << vd.word_str.back() << "'";
        for(size_t i=0;i<cols;i++) {
		std::cout << float_data[i] << " ";
	}
	std::cout << std::endl;
*/
	++pd;
        vd.num_samples++;
    }
    LOG_INFO << "skipped words = " << skipped_words << " (" 
	<< float(skipped_words)/float(rows)*100.0 << "%)";
    vd.num_features = cols;
    delete[] float_data;
    return vd;
}


po::variables_map parse_cmdargs(int argc, char const* argv[])
{

    po::variables_map vm;
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help,h", "produce help message")
        ("vec-file,v",po::value<std::string>()->required(), "word vector file")
        ("clusters,c",po::value<uint32_t>()->required(), "desired number of clusters")
        ("max-word-len,w",po::value<uint32_t>()->default_value(32), "maximum word len")
        ("device-list,d",po::value<std::string>()->required(), "GPU list: 0,1,2 ");
    // clang-format on
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            exit(EXIT_SUCCESS);
        }
        po::notify(vm);
    } catch (const po::required_option& e) {
        std::cout << desc;
        std::cerr << "Missing required option: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (po::error& e) {
        std::cout << desc;
        std::cerr << "Error parsing cmdargs: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return vm;
}

uint32_t generate_device_mask(std::string dev_list)
{
    uint32_t mask = 0;
    std::stringstream ss(dev_list);
    std::string dev;
    while (std::getline(ss, dev, ',')) {
        auto dev_num = std::atoi(dev.c_str());
        mask = mask | 1U << (dev_num);
    }
    return mask;
}

int main(int argc, char const* argv[])
{
    logging::init();

    auto cmdargs = parse_cmdargs(argc, argv);
    auto word_vec_file = cmdargs["vec-file"].as<std::string>();
    auto num_clusters = cmdargs["clusters"].as<uint32_t>();
    auto device_list = cmdargs["device-list"].as<std::string>();
    auto max_word_len = cmdargs["max-word-len"].as<uint32_t>();
    auto device_mask = generate_device_mask(device_list);

    LOG_INFO << "device_list = " << device_list;
    LOG_INFO << "device_mask = " << device_mask;
    LOG_INFO << "num clusters = " << num_clusters;
    LOG_INFO << "max_word_len = " << max_word_len;

    auto init = kmcuda::init_methods.find("kmeans++")->second;
    auto metric = kmcuda::metrics.find("euclidean")->second;
    auto tolerance = 0.002f;
    auto yinyang = 0.0f;

    LOG_INFO << "init = kmeans++";
    LOG_INFO << "metric = euclidean";

    auto vec_data = read_vector_data(word_vec_file,max_word_len);

    size_t size_bytes = vec_data.dat.size() * sizeof(vec_data.dat[0]);
    LOG_INFO << "data size in MiB = "
             << float(size_bytes) / float(8 * 1024 * 1024);

    // cluster parameters
    uint32_t rand_seed = 1234;
    int32_t fp16x2 = 0;
    int32_t verbosity = 2; // 0: no output, 2: debug output

    LOG_INFO << "rand_seed = " << rand_seed;
    LOG_INFO << "num_features = " << vec_data.num_features;
    LOG_INFO << "num_samples = " << vec_data.num_samples;

    const float* input_samples = (const float*) vec_data.dat.data();
    std::vector<float> raw_out_centroids(num_clusters * vec_data.num_features);
    float* output_centroids = raw_out_centroids.data();
    std::vector<uint32_t> raw_out_assignments(vec_data.num_samples);
    uint32_t* output_assignments = raw_out_assignments.data();

    /*
        for(size_t i=0;i<vec_data.num_samples;i++) {
            auto sptr = input_samples + i*vec_data.num_features;
            std::cout << "("<<i<<") => ";
            for(size_t j=0;j<vec_data.num_features;j++)
                    std::cout << sptr[j] << " ";
            std::cout << std::endl;
        }
    */

    float avg_distance = 0.0;
    {
        cl_timer<> cluster_start("kmeans_cuda");
        auto res = kmeans_cuda(init,
                               nullptr,
                               tolerance,
                               yinyang,
                               metric,
                               vec_data.num_samples,
                               vec_data.num_features,
                               num_clusters,
                               rand_seed,
                               device_mask,
                               -1, // device_ptrs: If negative, input and output
                               // pointers are taken from host
                               fp16x2,
                               verbosity,
                               input_samples,
                               output_centroids,
                               output_assignments,
                               &avg_distance);

        std::cout << "Status: " << kmcuda::statuses.find(res)->second
                  << std::endl;

        // (2) output clusters
        std::unordered_multimap<uint32_t, std::string> clusters;
        for (size_t i = 0; i < raw_out_assignments.size(); i++) {
            size_t cid = raw_out_assignments[i];
            std::string word = vec_data.word_str[i];
            clusters.insert({ cid, word });
        }

        for (auto& x : clusters)
            std::cout << x.first << ": " << x.second << std::endl;

	
	for(size_t i=0;i<num_clusters;i++) {
	    std::cout << "CENTROID " << i << ": ";
	    for(size_t j=0;j<vec_data.num_features;j++) {
		std::cout << output_centroids[i*vec_data.num_features+j] << " ";
	    }
	    std::cout << std::endl;
	}	
    }
}
