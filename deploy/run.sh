#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
# if [ ! -d caddn_infer_model ]; then
#     wget 
#     tar xzf caddn_infer_model.tgz
# fi

# 3. run
./build/caddn_main