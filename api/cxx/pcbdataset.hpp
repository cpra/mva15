
// API interface.
// ~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology

#pragma once

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include <string>
#include <vector>
#include <unordered_map>

namespace pcbdataset
{
    /// An annotated PCB component.
    struct Annot
    {
        /// Occupied region.
        cv::RotatedRect rect;

        /// Scale factor.
        float scale;

        /// Label text.
        std::string text;

        /// Constructor.
        /// @param rect occupied region.
        /// @param scale scale factor.
        /// @param text label text.
        Annot(const cv::RotatedRect& rect, float scale, const std::string& text);

        /// Returns the size of the component in pixels.
        /// @param scaled whether to regard the scale factor.
        float sizePixels(bool scaled) const;

        /// Returns the size of the component in cm^2.
        /// @param scaled whether to regard the scale factor.
        float sizeCm2(bool scaled) const;

        /// Returns the aspect ratio (larger side length / smaller side length).
        float aspect() const;
    };



    /// A printed circuit board.
    class PCB
    {
    public :

        /// Root directory path.
        const boost::filesystem::path root;

        /// Scale factor.
        const float scale;

        /// Constructor.
        /// @param root root directory path (no trailing /).
        /// @param scale scale factor (1 = original size).
        PCB(const std::string& root, float scale);

        /// Returns the PCB ID.
        int id() const;

        /// Returns a vector of IDs for all available recordings.
        std::vector<int> recordings() const;

        /// Returns the image of the specified recording.
        /// @param rec desired recording.
        /// @see recordings().
        cv::Mat image(int rec) const;

        /// Returns the mask of the specified recording.
        /// @param rec desired recording.
        /// @see recordings().
        cv::Mat mask(int rec) const;

        /// Returns the mask of the specified recording,
        /// masked by the corresponding mask and cropped to remove background.
        /// @param rec desired recording.
        /// @see recordings().
        cv::Mat imageMasked(int rec);

        /// Returns a list of IC chips as a vector of Annot objects.
        /// @param rec desired recording.
        /// @param cropped whether to return coordinates for cropped images (see imageMasked()).
        /// @param size (min, max) size of returned ICs in cm^2, disregarding the scale factor (0 = all).
        /// @param aspect (min, max) aspect ratio of returned ICs (0 = all).
        /// @see recordings(), imageMasked().
        std::vector<Annot> ics(int rec, bool cropped, cv::Vec2f size, cv::Vec2f aspect);

    private :

        /// id -> image_path pairs.
        std::unordered_map<int, boost::filesystem::path> _recordings;

        /// id -> crop pairs (cropinfo cache).
        std::unordered_map<int, cv::Rect> _cache_cropinfo;

        /// id -> ics pairs (ics cache).
        std::unordered_map<int, std::vector<Annot> > _cache_ics;

        /// Return (and cache) information for auto cropping a PCB image.
        /// @param rec desired recording.
        /// @see recordings().
        cv::Rect _cropinfo(int rec);
    };



    class PCBDataset
    {
    public :

        /// Path to the dataset.
        const boost::filesystem::path root;

        /// Constructor.
        /// @param root path to the dataset.
        PCBDataset(const std::string& root);

        /// Returns the number of PCBs in the dataset.
        int numPCBs() const;

        /// Returns a sorted vector of IDs of all PCBs in the dataset.
        std::vector<int> pcbIDs() const;

        /// Returns the PCB with the given ID.
        /// @param id PCB ID
        /// @param scale scale factor (1 = original size).
        /// @see pcbIDs().
        PCB pcb(int id, float scale) const;

    private :

        /// ID -> PCB path pairs.
        std::unordered_map<int, boost::filesystem::path> _pcbPaths;
    };
}
