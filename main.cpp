#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/photo/photo.hpp>
#include "adaptive_manifold_filter.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ @0                   |      | Input image }"
        "{ o output             |      | Output image }"
        "{ j joint              |      | Image for joint filtering (optional) }"
        "{ s sigma_s            | 16.0 | Filter spatial standard deviation }"
        "{ r sigma_r            | 0.2  | Filter range standard deviation }"
        "{ t tree_height        | -1   | Height of the manifold tree (default = -1 : automatically computed) }"
        "{ i num_pca_iterations | 1    | Number of iterations to computed the eigenvector v1 }"
        "{ h help               |      | Print help message }"
    );

    if (cmd.has("help"))
    {
        cmd.about("This sample demonstrates adaptive manifold filter algorithm");
        cmd.printMessage();
        return 0;
    }

    const string inputImageName = cmd.get<string>(0);
    const string outputImageName = cmd.get<string>("output");
    const string jointImageName = cmd.get<string>("joint");
    const double sigma_s = cmd.get<double>("sigma_s");
    const double sigma_r = cmd.get<double>("sigma_r");
    const int tree_height = cmd.get<int>("tree_height");
    const int num_pca_iterations = cmd.get<int>("num_pca_iterations");

    if (!cmd.check())
    {
        cmd.printErrors();
        return -1;
    }

    if (inputImageName.empty())
    {
        cerr << "Missing input image" << endl;
        cmd.printMessage();
        return -1;
    }

    Mat img = imread(inputImageName);
    if (img.empty())
    {
        cerr << "Can't open image - " << inputImageName << endl;
        return -1;
    }

    Mat jointImg;
    if (!jointImageName.empty())
    {
        jointImg = imread(jointImageName);
        if (jointImg.empty())
        {
            cerr << "Can't open image - " << inputImageName << endl;
            return -1;
        }
    }

    AdaptiveManifoldFilter filter;
    filter.set("sigma_s", sigma_s);
    filter.set("sigma_r", sigma_r);
    filter.set("tree_height", tree_height);
    filter.set("num_pca_iterations", num_pca_iterations);

    Mat dst, tilde_dst;
    filter.apply(img, dst, tilde_dst, jointImg);

    TickMeter tm;
    tm.start();
    filter.apply(img, dst, tilde_dst, jointImg);
    tm.stop();
    cout << "Time : " << tm.getTimeMilli() << " ms" << endl;

    Mat nlm_dst;
    fastNlMeansDenoisingColored(img, nlm_dst, 10.0, 10.0);
    tm.reset(); tm.start();
    fastNlMeansDenoisingColored(img, nlm_dst, 10.0, 10.0);
    tm.stop();
    cout << "NLM : " << tm.getTimeMilli() << " ms" << endl;

    if (!outputImageName.empty())
    {
        const string::size_type dotPos = outputImageName.find_last_of('.');
        const string name = outputImageName.substr(0, dotPos);
        const string ext = outputImageName.substr(dotPos + 1);

        imwrite(outputImageName, dst);
        imwrite(name + "_tilde." + ext, tilde_dst);
    }

    imshow("Input", img);
    imshow("Output", dst);
    imshow("Tilde Output", tilde_dst);
    imshow("NLM", nlm_dst);
    waitKey();

    return 0;
}
