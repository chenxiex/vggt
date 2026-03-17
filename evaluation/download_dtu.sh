#!/bin/bash

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output) output="$2"; shift ;;
        --cache) cache="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

output="${output:-data}"
cache="${cache:-${output}}"

download_and_extract() 
{
    local url="$1"
    local output_path="$2"
    local extract_dir="$3"

    if [[ -d "${extract_dir}" ]]; then
        echo "Directory ${extract_dir} already exists. Skipping download and extraction."
        return
    fi
    
    mkdir -p "$(dirname "${output_path}")"
    if [[ -f "${output_path}" ]]; then
        echo "Resuming download ${url} to ${output_path}..."
    else
        echo "Downloading ${url} to ${output_path}..."
    fi
    curl -L -C - -o "${output_path}" "${url}"
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

mkdir -p "${cache}"

# https://github.com/JiayuYANG/CVP-MVSNet?tab=readme-ov-file#2-download-testing-dataset
download_and_extract "https://www.kaggle.com/api/v1/datasets/download/chenxiex/dtu-test-1200" "${cache}/dtu-test-1200.zip" "${output}/dtu-test-1200"

# https://github.com/alibaba/cascade-stereo/blob/master/CasMVSNet/README.md#training
download_and_extract "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip" "${cache}/dtu_depths_raw.zip" "${output}/dtu_depths_raw"

# https://roboimagedata.compute.dtu.dk/?page_id=36
download_and_extract "http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip" "${cache}/dtu_sample.zip" "${output}/dtu_sample"
download_and_extract "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip" "${cache}/dtu_points.zip" "${cache}/dtu_points"
cp -r "${cache}/dtu_points/Points/stl/"* "${output}/dtu_sample/SampleSet/MVS Data/Points/stl/"
rm -r "${cache}/dtu_points"

rm "${cache}/dtu-test-1200.zip" "${cache}/dtu_depths_raw.zip" "${cache}/dtu_sample.zip" "${cache}/dtu_points.zip"