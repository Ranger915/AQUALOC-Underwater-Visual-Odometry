// Dataset adapter and diagnostics only; ORB-SLAM3 estimates all motion and maps.
#include <System.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>

int main(int argc,char** argv) {
 if(argc!=7) {std::cerr<<"vocabulary settings manifest preprocessing output seed\n";return 2;}
 const std::string prep=argv[4],out=argv[5];
 if(prep!="raw" && prep!="fixed" && prep!="adaptive")return 2;
 cv::setNumThreads(1);cv::setRNGSeed(std::stoi(argv[6]));std::srand(std::stoi(argv[6]));
 std::ifstream manifest(argv[3]);if(!manifest)return 3;
 std::vector<std::string> paths;std::vector<long long> times;std::vector<int> raw;
 long long stamp;int decision;std::string path;
 while(manifest>>stamp>>decision>>path){times.push_back(stamp);raw.push_back(decision);paths.push_back(path);}
 if(paths.empty() || !manifest.eof())return 3;
 for(size_t i=1;i<times.size();++i)if(times[i]<=times[i-1])return 3;
 ORB_SLAM3::System slam(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,false);
 std::ofstream frames(out+"/online.csv");if(!frames)return 4;
 frames<<"frame,timestamp_ns,state,map_id,processing_ms,px,py,pz,qx,qy,qz,qw\n"<<std::setprecision(12);
 auto clahe=cv::createCLAHE(2.,cv::Size(8,8));
 for(size_t i=0;i<paths.size();++i) {
  auto start=std::chrono::steady_clock::now();cv::Mat im=cv::imread(paths[i],cv::IMREAD_GRAYSCALE);
  if(im.empty() || im.cols!=968 || im.rows!=608)return 5;
  if(prep=="fixed" || (prep=="adaptive" && !raw[i]))clahe->apply(im,im);
  const auto pose=slam.TrackMonocular(im,times[i]*1e-9).inverse();
  const auto q=pose.unit_quaternion();const auto p=pose.translation();
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  frames<<i<<','<<times[i]<<','<<slam.GetTrackingState()<<','<<slam.AtlasForExport()->GetCurrentMap()->GetId()<<','<<elapsed*1000<<','<<p.x()<<','<<p.y()<<','<<p.z()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<'\n';
  if(i%100==0){frames.flush();std::cout<<"AQUALOC_FRAME "<<i<<std::endl;}
  const double gap=i+1<times.size()?(times[i+1]-times[i])*1e-9:.05;
  if(elapsed<gap)std::this_thread::sleep_for(std::chrono::duration<double>(gap-elapsed));
 }
 frames.close();slam.Shutdown();auto wait=std::chrono::steady_clock::now();
 while(!slam.WorkersFinished()) {
  if(std::chrono::duration<double>(std::chrono::steady_clock::now()-wait).count()>300)return 6;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
 }
 std::ofstream maps(out+"/maps.csv");maps<<"map_id,keyframes\n";
 for(auto map:slam.AtlasForExport()->GetAllMaps()) {
  const int n=map->KeyFramesInMap();maps<<map->GetId()<<','<<n<<'\n';
  if(n)slam.SaveTrajectoryEuRoC(out+"/map_"+std::to_string(map->GetId())+".txt",map);
 }
 maps.close();std::ofstream done(out+"/complete.txt");done<<paths.size()<<'\n';
}
