"""
@Time : 2021/4/1418:33
@Auth : 周俊贤
@File ：data_utils.py.py
@DESCRIPTION:

"""
import hashlib
import json

from utils.utils import read_by_lines

enum_role = "环节"

def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()

def text_to_sents(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    paragraphs = text.split("\n")
    ret = []
    for para in paragraphs:
        if para == u"":
            continue
        sents = [u""]
        for s in para:
            sents[-1] += s
            if s in deliniter_symbols:
                sents.append(u"")
        if sents[-1] == u"":
            sents = sents[:-1]
        ret.extend(sents)
    return ret

def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if "B-{}".format(_type) not in labels:
            labels.extend(["B-{}".format(_type), "I-{}".format(_type)])
        return labels

    labels = []
    for line in read_by_lines(path):
        d_json = json.loads(line.strip())
        if model == "trigger":
            labels = label_add(labels, d_json["event_type"])
        elif model == "role":
            for role in d_json["role_list"]:
                if role["role"] == enum_role:
                    continue
                labels = label_add(labels, role["role"])
        elif model == "enum":
            for role in d_json["role_list"]:
                if role["role"] == enum_role:
                    labels = role["enum_items"]

    labels.append("O")
    tags = []
    for index, label in enumerate(labels):
        tags.append("{}\t{}".format(index, label))
    if model == "enum":
        tags = tags[:-1]
    return tags

def docs_data_process(path):
    """docs_data_process"""
    lines = read_by_lines(path)
    sentences = []
    for line in lines:
        d_json = json.loads(line)
        sentences.extend(marked_doc_2_sentence(d_json))
    sentences = [json.dumps(s, ensure_ascii=False) for s in sentences]
    return sentences

def data_process(path, model="trigger", type="", is_predict=False):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = "B-" if i == start else "I-"
            data[i] = "{}{}".format(suffix, _type)
        return data

    output = ["text_a\tlabel"]

    for line in read_by_lines(path):
        d_json = json.loads(line.strip())
        _id = d_json["id"]
        text_a = [
            "，" if t == " " or t == "\n" or t == "\t" else t
            for t in list(d_json["text"].lower())
        ]
        if model == "trigger":
            labels = ["O"] * len(text_a)
            if is_predict:
                output.append("{}\t{}".format('\002'.join(text_a), '\002'.join(labels)))
            else:
                if (len(d_json.get("event_list", [])) == 0) and (type != "duee1"):
                    continue
                for event in d_json.get("event_list", []):
                    event_type = event["event_type"]
                    start = event["trigger_start_index"]
                    trigger = event["trigger"]
                    # 这步不会出现嵌套实体？
                    labels = label_data(labels,
                                        start,
                                        len(trigger),
                                        event_type)
                output.append("{}\t{}".format('\002'.join(text_a), '\002'.join(labels)))
        elif model == "role":
            labels = ["O"] * len(text_a)
            if is_predict:
                output.append("{}\t{}".format('\002'.join(text_a), '\002'.join(labels)))
            else:
                for event in d_json.get("event_list", []):
                    for arg in event["arguments"]:
                        role_type = arg["role"]
                        if role_type == enum_role:
                            continue
                        argument = arg["argument"]
                        start = arg["argument_start_index"]
                        labels = label_data(labels,
                                            start,
                                            len(argument),
                                            role_type)
                    output.append("{}\t{}".format('\002'.join(text_a), '\002'.join(labels)))

    return output

def enum_data_process(path, is_predict=False):
    """enum_data_process"""
    output = ["label\ttext_a"]
    for line in read_by_lines(path):
        d_json = json.loads(line)
        text = d_json["text"].lower().replace("\t", " ")
        if is_predict:
            output.append("{}\t{}".format("占位符", text))
            continue
        if len(d_json.get("event_list", [])) == 0:
            continue
        label = None
        for event in d_json["event_list"]:
            if event["event_type"] != "公司上市":
                continue
            for argument in event["arguments"]:
                role_type = argument["role"]
                if role_type == enum_role:
                    label = argument["argument"]
        if label:
            output.append("{}\t{}".format(label, text))
    return output

def marked_doc_2_sentence(doc):
    """marked_doc_2_sentence"""

    def argument_in_sent(sent, argument_list, trigger):
        """argument_in_sent"""
        trigger_start = sent.find(trigger)
        if trigger_start < 0:
            return trigger_start, [], None
        new_arguments, enum_argument = [], None
        for argument in argument_list:
            word = argument["argument"]
            role_type = argument["role"]
            if role_type == enum_role:
                # special
                enum_argument = argument
                continue
            start = sent.find(word)
            if start < 0:
                continue
            new_arguments.append({
                "role": role_type,
                "argument": word,
                "argument_start_index": start
            })
        return trigger_start, new_arguments, enum_argument

    # doc为一个dict：有'text', 'event_list', 'id', 'title'四个字段
    title = doc["title"]
    text = doc["text"]
    sents = text_to_sents(text)
    exist_sents, sent_mapping_event, sents_order = set(), {}, []
    step = 3  # 这个step的设置要好好探究一下
    batch_sents = [sents[i:i + step] for i in range(0, len(sents), step)]
    if len(title) > 0:
        batch_sents = [[title]] + batch_sents
    for batch in batch_sents:
        b_sent = " ".join(batch).replace("\n", " ").replace(
            "\r\n", " ").replace("\r", " ").replace("\t", " ")
        if b_sent in sent_mapping_event:
            continue
        sent_id = cal_md5(b_sent.encode("utf-8"))
        sent_mapping_event[b_sent] = {
            "id": doc["id"],
            "sent_id": sent_id,
            "text": b_sent
        }
        sents_order.append(b_sent)

    # doc["event_list"]为一个dict：有'trigger'、'event_type'、'arguments'
    for event in doc.get("event_list", []):
        cur_sent, trigger_start, arguments, enum_argument = "", -1, [], None
        for sent in sents_order:
            tri_start, argus, enum_arg = argument_in_sent(
                sent, event["arguments"], event["trigger"])
            if tri_start < 0:
                continue
            if len(argus) > len(arguments):
                cur_sent, trigger_start, arguments = sent, tri_start, argus
            if enum_arg:
                enum_argument = enum_arg
        if trigger_start >= 0 and len(arguments) > 0:
            # add enum 2 event
            if enum_argument:
                arguments.append(enum_argument)
            if "event_list" not in sent_mapping_event[cur_sent]:
                sent_mapping_event[cur_sent]["event_list"] = []
            new_event = {
                "arguments": arguments,
                "event_type": event["event_type"],
                "trigger": event["trigger"],
                "trigger_start_index": trigger_start
            }
            sent_mapping_event[cur_sent]["event_list"].append(new_event)
    return sent_mapping_event.values()
