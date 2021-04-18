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
"""duee finance dataset proces"""
import os

from data.data_utils import schema_process, data_process, docs_data_process, enum_data_process
from utils.utils import write_by_lines

if __name__ == "__main__":
    # schema process
    print("\n=================DUEE FINANCE DATASET==============")
    conf_dir = "./conf/DuEE-Fin"
    schema_path = "{}/event_schema.json".format(conf_dir)
    tags_trigger_path = "{}/trigger_tag.dict".format(conf_dir)
    tags_role_path = "{}/role_tag.dict".format(conf_dir)
    tags_enum_path = "{}/enum_tag.dict".format(conf_dir)
    print("\n=================start schema process==============")
    print('input path {}'.format(schema_path))
    tags_trigger = schema_process(schema_path, "trigger")
    write_by_lines(tags_trigger_path, tags_trigger)
    print("save trigger tag {} at {}".format(len(tags_trigger), tags_trigger_path))
    tags_role = schema_process(schema_path, "role")
    write_by_lines(tags_role_path, tags_role)
    print("save trigger tag {} at {}".format(len(tags_role), tags_role_path))
    tags_enum = schema_process(schema_path, "enum")
    write_by_lines(tags_enum_path, tags_enum)
    print("save enum enum tag {} at {}".format(len(tags_enum), tags_enum_path))
    print("=================end schema process===============")

    # data process
    data_dir = "./data/DuEE-Fin"
    sentence_dir = "{}/sentence".format(data_dir)
    trigger_save_dir = "{}/trigger".format(data_dir)
    role_save_dir = "{}/role".format(data_dir)
    enum_save_dir = "{}/enum".format(data_dir)

    print("\n=================start data process==============")
    print("\n********** start document process **********")
    if not os.path.exists(sentence_dir):
        os.makedirs(sentence_dir)
    # train_sent是一个列表，每个元素为一个字典，字典可能包含（id、sent_id、text三个字段），也可能包含（id、sent_id、text、event_list四个字段）
    train_sent = docs_data_process("{}/duee_fin_train.json".format(data_dir))
    write_by_lines("{}/train.json".format(sentence_dir), train_sent)
    dev_sent = docs_data_process("{}/duee_fin_dev.json".format(data_dir))
    write_by_lines("{}/dev.json".format(sentence_dir), dev_sent)
    test_sent = docs_data_process("{}/duee_fin_test1.json".format(data_dir))
    write_by_lines("{}/test.json".format(sentence_dir), test_sent)
    print("train {} dev {} test {}".format(len(train_sent), len(dev_sent), len(test_sent)))
    print("********** end document process **********")

    print("\n********** start sentence process **********")
    #
    print("\n----trigger------for dir {} to {}".format(sentence_dir, trigger_save_dir))
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)
    train_tri = data_process("{}/train.json".format(sentence_dir), "trigger", type="duee_fin")
    write_by_lines("{}/train.tsv".format(trigger_save_dir), train_tri)
    dev_tri = data_process("{}/dev.json".format(sentence_dir), "trigger", type="duee_fin")
    write_by_lines("{}/dev.tsv".format(trigger_save_dir), dev_tri)
    test_tri = data_process("{}/test.json".format(sentence_dir), "trigger", type="duee_fin", is_predict=True)
    write_by_lines("{}/test.tsv".format(trigger_save_dir), test_tri)
    print("train {} dev {} test {}".format(len(train_tri), len(dev_tri), len(test_tri)))
    #
    print("\n----role------for dir {} to {}".format(sentence_dir, role_save_dir))
    if not os.path.exists(role_save_dir):
        os.makedirs(role_save_dir)
    train_role = data_process("{}/train.json".format(sentence_dir), "role", type="duee_fin")
    write_by_lines("{}/train.tsv".format(role_save_dir), train_role)
    dev_role = data_process("{}/dev.json".format(sentence_dir), "role", type="duee_fin")
    write_by_lines("{}/dev.tsv".format(role_save_dir), dev_role)
    test_role = data_process("{}/test.json".format(sentence_dir), "role", type="duee_fin", is_predict=True)
    write_by_lines("{}/test.tsv".format(role_save_dir), test_role)
    print("train {} dev {} test {}".format(len(train_role), len(dev_role), len(test_role)))
    #
    print("\n----enum------for dir {} to {}".format(sentence_dir, enum_save_dir))
    if not os.path.exists(enum_save_dir):
        os.makedirs(enum_save_dir)
    trian_enum = enum_data_process("{}/train.json".format(sentence_dir))
    write_by_lines("{}/train.tsv".format(enum_save_dir), trian_enum)
    dev_enum = enum_data_process("{}/dev.json".format(sentence_dir))
    write_by_lines("{}/dev.tsv".format(enum_save_dir), dev_enum)
    test_enum = enum_data_process("{}/test.json".format(sentence_dir), is_predict=True)
    write_by_lines("{}/test.tsv".format(enum_save_dir), test_enum)
    #
    print("train {} dev {} test {}".format(len(trian_enum), len(dev_enum), len(test_enum)))
    print("********** end sentence process **********")
    print("=================end data process==============")
