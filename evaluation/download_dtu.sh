#!/bin/bash

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output) output="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

output="${output:-data}"

download_and_extract() 
{
    local url="$1"
    local output_path="$2"
    local extract_dir="$3"

    curl -L -o "${output_path}" "${url}"
    mkdir -p "${extract_dir}"
    unzip -o "${output_path}" -d "${extract_dir}"

    # Handle potential nested directory
    local nested_dir="${extract_dir}/$(basename "${output_path}" .zip)"
    if [[ -d "${nested_dir}" ]]; then
        shopt -s dotglob nullglob
        mv -f "${nested_dir}"/* "${extract_dir}/"
        shopt -u dotglob nullglob
        rmdir "${nested_dir}" 2>/dev/null || true
    fi
}

# https://github.com/JiayuYANG/CVP-MVSNet?tab=readme-ov-file#2-download-testing-dataset
download_and_extract "https://modelscope.cn/datasets/anlorsp/dtu-test-1200/resolve/master/dtu-test-1200.zip" "${output}/dtu-test-1200.zip" "${output}/dtu-test-1200"

# https://github.com/alibaba/cascade-stereo/blob/master/CasMVSNet/README.md#training
download_and_extract "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip" "${output}/dtu_depths_raw.zip" "${output}/dtu_depths_raw"

# https://roboimagedata.compute.dtu.dk/?page_id=36
download_and_extract "http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip" "${output}/dtu_sample.zip" "${output}/dtu_sample"
download_and_extract "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip" "${output}/dtu_points.zip" "${output}/dtu_points"
cp -r "${output}/dtu_points/Points/stl/"* "${output}/dtu_sample/SampleSet/MVS Data/Points/stl/"