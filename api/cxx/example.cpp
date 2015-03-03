
// Demonstrates the use of the DSLR dataset C++ API.
// ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

#include "pcbdataset.hpp"

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    using std::cout;
    using std::cerr;
    using std::endl;

    namespace po = boost::program_options;

    using namespace pcbdataset;

    // parse args

    std::string root, icszStr, icasStr;
    int pid, rid;
    float scale;

    po::options_description desc("Runtime Arguments");
    desc.add_options()
        ("help", "Print this message and quit.")
        ("root", po::value<std::string>(&root)->default_value(""), "Path to the dataset.")
        ("pcb", po::value<int>(&pid)->default_value(1), "ID of the PCB to show.")
        ("rec", po::value<int>(&rid)->default_value(1), "ID of the recording to show.")
        ("scale", po::value<float>(&scale)->default_value(1), "Scale factor.")
        ("icsz", po::value<std::string>(&icszStr)->default_value("0,0"), "(min, max) size of returned ICs in cm^2 (0 = no restriction).")
        ("icas", po::value<std::string>(&icasStr)->default_value("0,0"), "(min, max) aspect ratio of returned ICs (0 = no restriction).");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help")) {
        cout << desc << endl;
        return 0;
    }

    if(root.empty()) {
        cerr << "--root must be specified" << endl;
        return 1;
    }

    std::vector<std::string> split;
    boost::split(split, icszStr, boost::is_any_of(","));
    if(split.size() != 2) {
        cerr << "--icsz has an invalid format" << endl;
        return 1;
    }
    cv::Vec2f icsz(std::stof(split[0]), std::stof(split[1]));

    split.clear();
    boost::split(split, icasStr, boost::is_any_of(","));
    if(split.size() != 2) {
        cerr << "--icas has an invalid format" << endl;
        return 1;
    }
    cv::Vec2f icas(std::stof(split[0]), std::stof(split[1]));

    cout << icsz << " / " << icas << endl;

    // show data

    PCBDataset dataset(root);

    cout << "Dataset contains images of " << dataset.numPCBs() << " PCBs" << endl;
    cout << " IDs: ";
    for(int pid : dataset.pcbIDs())
        cout << pid << " ";
    cout << endl;

    PCB pcb = dataset.pcb(pid, scale);

    cout << "Loaded PCB " << pcb.id() << ", available recordings: ";
    for(int r : pcb.recordings())
        cout << r << " ";
    cout << endl;

    std::vector<Annot> ics = pcb.ics(rid, true, icsz, icas);
    cout << "PCB contains " << ics.size() << " ICs" << endl;

    cv::Mat im = pcb.imageMasked(1);

    for(auto& an : ics) {
        cv::Point2f rpts[4]; an.rect.points(rpts);
        for(int j = 0; j < 4; j++)
           cv::line(im, rpts[j], rpts[(j+1)%4], cv::Scalar(0, 255, 0), 2, 8);
    }

    cv::imshow("PCB", im);
    cv::waitKey(0);

    return 0;
}
