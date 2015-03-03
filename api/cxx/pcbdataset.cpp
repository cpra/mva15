
// API implementation.
// ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

#include "pcbdataset.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <fstream>

namespace pcbdataset
{
    Annot::Annot(const cv::RotatedRect& rect, float scale, const std::string& text) :
        rect(rect),
        scale(scale),
        text(text)
    { }

    float Annot::sizePixels(bool scaled) const
    {
        float sz = rect.size.area();
        if(! scaled)
            sz /= scale;

        return sz;
    }

    float Annot::sizeCm2(bool scaled) const
    {
        float sz = (rect.size.width / 87.4) * (rect.size.height / 87.4);
        if(! scaled)
            sz /= scale;

        return sz;
    }

    float Annot::aspect() const
    {
        return std::max(rect.size.width, rect.size.height) / std::min(rect.size.width, rect.size.height);
    }



    PCB::PCB(const std::string& root, float scale) :
        root(root),
        scale(scale)
    {
        if(! boost::filesystem::is_directory(root))
            throw std::invalid_argument("Root path is not a directory.");

        if(root.substr(root.size()-1, 1) == "/")
            throw std::invalid_argument("Root path must not end with /");

        if(scale <= 0 || scale > 2)
            throw std::invalid_argument("Scale must be > 0 and <= 2.");

        boost::filesystem::directory_iterator itr(root), endItr;
        BOOST_FOREACH(const boost::filesystem::path& p, std::make_pair(itr, endItr)) {
            if(boost::filesystem::is_regular_file(p)) {
                std::string fname = p.filename().string();
                if(fname.size() > 7 && fname.substr(0, 3) == "rec" && fname.substr(fname.size()-4, 4) == ".jpg") {
                    std::string rid = fname.substr(3, std::string::npos); rid = rid.substr(0, rid.size()-4);
                    _recordings.insert({ std::stoi(rid), p });
                }
            }
        }
    }

    int PCB::id() const
    {
        return std::stoi(root.filename().string().substr(3, std::string::npos));
    }

    std::vector<int> PCB::recordings() const
    {
        std::vector<int> ret;
        for(const auto& r : _recordings)
            ret.push_back(r.first);

        return ret;
    }

    cv::Mat PCB::image(int rec) const
    {
        if(_recordings.count(rec) == 0)
            throw std::invalid_argument("Recording does not exist.");

        cv::Mat image = cv::imread(_recordings.at(rec).string(), cv::IMREAD_UNCHANGED);
        if(image.rows == 0)
            throw std::runtime_error("Could not load the image.");

        if(scale != 1)
            cv::resize(image, image, cv::Size(0, 0), scale, scale);

        return image;
    }

    cv::Mat PCB::mask(int rec) const
    {
        if(_recordings.count(rec) == 0)
            throw std::invalid_argument("Recording does not exist.");

        boost::filesystem::path impath = _recordings.at(rec);
        std::stringstream ss; ss << "rec" << rec << "-mask.png";
        impath = impath.parent_path() / ss.str();

        if(! boost::filesystem::is_regular_file(impath))
            throw std::runtime_error("Mask file does not exist.");

        cv::Mat image = cv::imread(impath.string(), cv::IMREAD_GRAYSCALE);
        if(image.rows == 0)
            throw std::runtime_error("Could not load the image.");

        if(scale != 1)
            cv::resize(image, image, cv::Size(0, 0), scale, scale);

        return image;
    }

    cv::Mat PCB::imageMasked(int rec)
    {
        cv::Mat im = image(rec);
        cv::Mat msk = mask(rec);

        im.setTo(0, msk == 0);
        cv::Rect ci = _cropinfo(rec);

        return im(ci);
    }

    std::vector<Annot> PCB::ics(int rec, bool cropped, cv::Vec2f size, cv::Vec2f aspect)
    {
        if(_cache_ics.count(rec) == 1)
            return _cache_ics[rec];

        if(_recordings.count(rec) == 0)
            throw std::invalid_argument("Recording does not exist.");

        boost::filesystem::path fpath = _recordings.at(rec);
        std::stringstream ss; ss << "rec" << rec << "-annot.txt";
        fpath = fpath.parent_path() / ss.str();

        if(! boost::filesystem::is_regular_file(fpath))
            throw std::runtime_error("Annotation file does not exist.");

        std::vector<Annot> ret;

        std::string line;
        std::ifstream file(fpath.string());

        while(std::getline(file, line)) {
            std::vector<std::string> split;
            boost::split(split, line, boost::is_any_of(" "));

            if(split.size() < 5)
                throw std::runtime_error("Invalid line encountered while parsing file.");

            cv::RotatedRect rr(
                cv::Point2f(std::stof(split[0]), std::stof(split[1])),
                cv::Size2f(std::stof(split[2]), std::stof(split[3])),
                std::stof(split[4])
            );

            Annot tmp(rr, 1.0, "");

            if(size(0) > 0 && tmp.sizeCm2(false) < size(0))
                continue;

            if(size(1) > 0 && tmp.sizeCm2(false) > size(1))
                continue;

            if(aspect(0) > 0 && tmp.aspect() < aspect(0))
                continue;

            if(aspect(1) > 0 && tmp.aspect() > aspect(1))
                continue;

            if(scale != 1) {
                rr.center.x *= scale;
                rr.center.y *= scale;
                rr.size.width *= scale;
                rr.size.height *= scale;
            }

            if(cropped) {
                cv::Rect ci = _cropinfo(rec);
                rr.center.x -= ci.x;
                rr.center.y -= ci.y;
            }

            std::stringstream ss;
            for(size_t i = 5; i < split.size(); i++) {
                if(! split[i].empty()) {
                    ss << split[i];
                    if(i < split.size()-1)
                        ss << " ";
                }
            }

            ret.push_back(Annot(rr, scale, ss.str()));
        }

        file.close();

        _cache_ics.insert({ rec, std::move(ret) });

        return _cache_ics[rec];
    }

    cv::Rect PCB::_cropinfo(int rec)
    {
        if(_cache_cropinfo.count(rec) == 1)
            return _cache_cropinfo[rec];

        cv::Mat im = mask(rec);

        std::vector<std::vector<cv::Point> > cnt;
        cv::findContours(im, cnt, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        std::sort(cnt.begin(), cnt.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            cv::RotatedRect ra = cv::minAreaRect(a);
            cv::RotatedRect rb = cv::minAreaRect(b);

            return ra.size.area() < rb.size.area();
        });

        _cache_cropinfo.insert({ rec, cv::boundingRect(cnt[cnt.size()-1]) });

        return _cache_cropinfo[rec];
    }



    PCBDataset::PCBDataset(const std::string& root) : root(root)
    {
        if(! boost::filesystem::is_directory(root))
            throw std::invalid_argument("Root path is not a directory.");

        if(root.substr(root.size()-1, 1) == "/")
            throw std::invalid_argument("Root path must not end with /");

        boost::filesystem::directory_iterator itr(root), endItr;
        BOOST_FOREACH(const boost::filesystem::path& p, std::make_pair(itr, endItr)) {
            if(boost::filesystem::is_directory(p)) {
                std::string fname = p.filename().string();
                if(fname.size() > 3 && fname.substr(0, 3) == "pcb")
                    _pcbPaths.insert({ std::stoi(fname.substr(3, std::string::npos)), p });
            }
        }

        if(_pcbPaths.empty())
            throw std::runtime_error("Specified path contains no PCB directories.");
    }

    int PCBDataset::numPCBs() const
    {
        return _pcbPaths.size();
    }

    std::vector<int> PCBDataset::pcbIDs() const
    {
        std::vector<int> ret;
        for(const auto& p : _pcbPaths)
            ret.push_back(p.first);

        std::sort(ret.begin(), ret.end());

        return ret;
    }

    PCB PCBDataset::pcb(int id, float scale) const
    {
        if(_pcbPaths.count(id) == 0)
            throw std::invalid_argument("PCB does not exist.");

        return PCB(_pcbPaths.at(id).string(), scale);
    }
}
