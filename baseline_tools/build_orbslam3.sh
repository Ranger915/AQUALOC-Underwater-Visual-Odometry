#!/usr/bin/env bash
set -euo pipefail
work=/opt/aqualoc
adapter=/mnt/c/Users/Raaghav/Documents/ChatGPT/Deep_sea/baseline_tools
cd "$work"
if [ ! -d Pangolin ]; then git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git; fi
git -C Pangolin rev-parse HEAD > "$adapter/pangolin_commit.txt"
cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF
cmake --build Pangolin/build -j2
cmake --install Pangolin/build
ldconfig
cd ORB_SLAM3
test "$(git rev-parse HEAD)" = 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
python3 - <<'PY'
from pathlib import Path
p=Path('CMakeLists.txt');s=p.read_text().replace('-std=c++11','-std=c++14')
if 'mono_aqualoc' not in s:s+='\nadd_executable(mono_aqualoc /mnt/c/Users/Raaghav/Documents/ChatGPT/Deep_sea/baseline_tools/mono_aqualoc.cc)\ntarget_link_libraries(mono_aqualoc ORB_SLAM3)\nset_target_properties(mono_aqualoc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)\n'
if p.read_text()!=s:p.write_text(s)
p=Path('include/System.h');s=p.read_text()
if 'AtlasForExport' not in s:s=s.replace('    int GetTrackingState();','    Atlas* AtlasForExport() { return mpAtlas; }\n    bool WorkersFinished();\n    int GetTrackingState();')
s=s.replace('bool WorkersFinished() { return mpLocalMapper->isFinished() && mpLoopCloser->isFinished() && !mpLoopCloser->isRunningGBA(); }','bool WorkersFinished();')
if p.read_text()!=s:p.write_text(s)
p=Path('src/System.cc');s=p.read_text()
if 'System::WorkersFinished()' not in s:s+='\nbool ORB_SLAM3::System::WorkersFinished() { return mpLocalMapper->isFinished() && mpLoopCloser->isFinished() && !mpLoopCloser->isRunningGBA(); }\n'
if p.read_text()!=s:p.write_text(s)
PY
for lib in DBoW2 g2o; do
 cmake -S "Thirdparty/$lib" -B "Thirdparty/$lib/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_STANDARD_REQUIRED=ON
 cmake --build "Thirdparty/$lib/build" -j2
done
test -f Vocabulary/ORBvoc.txt || tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mono_aqualoc -j2
git diff > "$adapter/upstream_adapter.patch"
sha256sum Vocabulary/ORBvoc.txt lib/libORB_SLAM3.so Thirdparty/g2o/lib/libg2o.so Thirdparty/DBoW2/lib/libDBoW2.so /usr/local/lib/libpangolin.so Examples/Monocular/mono_aqualoc > "$adapter/build_sha256.txt"
cat build/CMakeFiles/ORB_SLAM3.dir/flags.make Thirdparty/g2o/build/CMakeFiles/g2o.dir/flags.make > "$adapter/build_flags.txt"
dpkg-query -W > "$adapter/ubuntu_packages.txt"
