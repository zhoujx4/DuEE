# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""duee 1.0 dataset process"""
import os

from data.data_utils import schema_process, data_process
from utils.utils import write_by_lines

if __name__ == "__main__":
    print("\n=================DUEE 1.0 DATASET==============")
    conf_dir = "./conf/DuEE1.0"
    schema_path = "{}/event_schema.json".format(conf_dir)
    tags_trigger_path = "{}/trigger_tag.dict".format(conf_dir)
    tags_role_path = "{}/role_tag.dict".format(conf_dir)
    print("\n=================start schema process==============")
    print('input path {}'.format(schema_path))
    tags_trigger = schema_process(schema_path, "trigger")
    write_by_lines(tags_trigger_path, tags_trigger)
    print("save trigger tag {} at {}".format(len(tags_trigger), tags_trigger_path))
    tags_role = schema_process(schema_path, "role")
    write_by_lines(tags_role_path, tags_role)
    print("save trigger tag {} at {}".format(len(tags_role), tags_role_path))
    print("=================end schema process===============")

    # data process
    data_dir = "./data/DuEE1.0"
    trigger_save_dir = "{}/trigger".format(data_dir)
    role_save_dir = "{}/role".format(data_dir)

    print("\n=================start sentence process==============")
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)
    if not os.path.exists(role_save_dir):
        os.makedirs(role_save_dir)

    print("\n----trigger------for dir {} to {}".format(data_dir, trigger_save_dir))
    train_tri = data_process("{}/duee_train.json".format(data_dir), "trigger", type="duee1")
    write_by_lines("{}/train.tsv".format(trigger_save_dir), train_tri)
    dev_tri = data_process("{}/duee_dev.json".format(data_dir), "trigger", type="duee1")
    write_by_lines("{}/dev.tsv".format(trigger_save_dir), dev_tri)
    test_tri = data_process("{}/duee_test1.json".format(data_dir), "trigger", type="duee1")
    write_by_lines("{}/test.tsv".format(trigger_save_dir), test_tri)
    print("train {} dev {} test {}".format(
        len(train_tri), len(dev_tri), len(test_tri)))

    print("\n----role------for dir {} to {}".format(data_dir, role_save_dir))
    train_role = data_process("{}/duee_train.json".format(data_dir), "role", type="duee1")
    write_by_lines("{}/train.tsv".format(role_save_dir), train_role)
    dev_role = data_process("{}/duee_dev.json".format(data_dir), "role", type="duee1")
    write_by_lines("{}/dev.tsv".format(role_save_dir), dev_role)
    test_role = data_process("{}/duee_test1.json".format(data_dir), "role", type="duee1", is_predict=True)
    write_by_lines("{}/test.tsv".format(role_save_dir), test_role)
    print("train {} dev {} test {}".format(len(train_role), len(dev_role), len(test_role)))
    print("=================end schema process==============")
