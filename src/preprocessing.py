# Yoochoose Data: https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
# Diginetica Data: https://drive.google.com/file/d/0B7XZSACQf0KdenRmMk8yVUU5LWc/
# Beauty Data: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/

import argparse
import json
import logging as log
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from tqdm.auto import tqdm


def read_file(filename, header=False):
    with open(filename, "r") as f:
        file_content = f.readlines()
    return file_content if not header else file_content[1:]


def sort_events(events):
    return sorted(events, key=lambda event: event["ts"])


def create_sessions(events, dataset_name, event_type="clicks"):
    sessions = dict()
    for event in tqdm(events):
        if dataset_name == "diginetica":
            if event_type == "clicks":
                sid, _uid, aid, timeframe, eventdate = event.strip().split(";")
            elif event_type == "orders":
                sid, _uid, timeframe, eventdate, _order_number, aid = event.strip().split(";")
            ts = (datetime.strptime(eventdate, '%Y-%m-%d') + timedelta(milliseconds=int(timeframe))).timestamp()
        elif dataset_name == "yoochoose":
            if event_type == "clicks":
                sid, ts, aid, _cat = event.strip().split(",")
            elif event_type == "orders":
                sid, ts, aid, _price, _quantity = event.strip().split(",")
            ts = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
        if not sid in sessions:
            sessions[sid] = list()
        sessions[sid].append({"aid": aid, "ts": ts, "type": event_type})
    sessions = [{"session": sid, "events": sort_events(events)} for sid, events in sessions.items()]
    return sessions


def sort_sessions(sessions):
    return sorted(sessions, key=lambda x: x["events"][0]["ts"])


def filter_short_sessions(sessions, min_session_len=2):
    return [session for session in tqdm(sessions) if len(session["events"]) >= min_session_len]


def get_aid_support(sessions):
    aid_support = {}
    for session in sessions:
        for event in session["events"]:
            aid = event["aid"]
            if aid in aid_support:
                aid_support[aid] += 1
            else:
                aid_support[aid] = 1
    return aid_support


def filter_low_aid_support(sessions, min_aid_support=5):
    aid_support = get_aid_support(sessions)
    for session in tqdm(sessions):
        session["events"] = list(filter(lambda event: aid_support[event["aid"]] >= min_aid_support, session["events"]))
    return sessions


def get_session_lengths(sessions):
    return {session["session"]: len(session["events"]) for session in sessions}


def filter_low_aid_and_sessions(sessions, min_aid_support, min_session_len):
    session_lengths = get_session_lengths(sessions)
    aid_support = get_aid_support(sessions)
    filtered_sessions = list()
    for session in tqdm(sessions):
        if session_lengths[session["session"]] >= min_session_len:
            session["events"] = list(filter(lambda event: aid_support[event["aid"]] >= min_aid_support, session["events"]))
            if len(session["events"]) > 0:
                filtered_sessions.append(session)
    return filtered_sessions


def apply_session_filtering(sessions, min_session_len=2, min_aid_support=5):
    sessions = filter_short_sessions(sessions, min_session_len)
    sessions = filter_low_aid_support(sessions, min_aid_support)
    return filter_short_sessions(sessions, min_session_len)


def train_test_split(sessions, dataset_name, split_seconds, split_idx):
    max_date = max([session["events"][0]["ts"] for session in sessions])
    if dataset_name == "diginetica":
        max_date = datetime.fromtimestamp(int(max_date)).strftime('%Y-%m-%d')
        max_date = time.mktime(time.strptime(max_date, '%Y-%m-%d'))
    splitdate = max_date - split_seconds
    train_sessions = filter(lambda session: session["events"][split_idx]["ts"] < splitdate, sessions)
    test_sessions = filter(lambda session: session["events"][split_idx]["ts"] >= splitdate, sessions)
    return (list(train_sessions), list(test_sessions))


def filter_aids(train_aids, test_sessions):
    test_aids = [event["aid"] for session in test_sessions for event in session["events"]]
    aids_to_remove = set(test_aids).difference(set(train_aids))
    for session in test_sessions:
        session["events"] = [event for event in session["events"] if not event["aid"] in aids_to_remove]
    return test_sessions


def create_aid_to_idx(train_aids):
    aid_to_idx = dict()
    aid_counter = 1
    for aid in tqdm(train_aids):
        if not aid in aid_to_idx:
            aid_to_idx[aid] = aid_counter
            aid_counter += 1
    return aid_to_idx


def remap_indices(sessions, aid_to_idx):
    num_click_events = 0
    num_order_events = 0
    num_sessions = 0
    for session in tqdm(sessions):
        for event in session["events"]:
            event["aid"] = aid_to_idx[event["aid"]]
            if event["type"] == "clicks":
                num_click_events += 1
            elif event["type"] == "orders":
                num_order_events += 1
        num_sessions += 1
    return sessions, num_sessions, num_click_events, num_order_events


def write_file(sessions, filename):
    with open(filename, "w") as f:
        for s in tqdm(sessions):
            f.write(json.dumps(s) + "\n")


def merge_clicks_and_orders(click_sessions, order_sessions):
    order_dict = {session["session"]: session["events"] for session in order_sessions}
    for session in tqdm(click_sessions):
        session["events"] = session["events"] + order_dict.get(session["session"], [])
        session["events"] = sort_events(session["events"])
    return click_sessions


def write_stats(num_items,
                num_train_sessions,
                num_train_click_events,
                num_train_order_events,
                num_test_sessions=None,
                num_test_click_events=None,
                num_test_order_events=None,
                filename=None):
    stats = {
        "train": {
            "num_sessions": num_train_sessions,
            "num_click_events": num_train_click_events,
            "num_order_events": num_train_order_events,
            "num_events": num_train_click_events + num_train_order_events
        },
        "num_items": num_items,
        "test": {
            "num_sessions": num_test_sessions,
            "num_click_events": num_test_click_events,
            "num_order_events": num_test_order_events,
            "num_events": num_test_click_events + num_test_order_events
        }
    }
    with open(filename, "w") as f:
        f.write(json.dumps(stats))


def run_preprocessing(config, data_dir):
    dataset_name = config["dataset_name"]
    click_events = read_file(config["data_file_clicks"], header=config["header"])
    log.info(f"Read {len(click_events)} events from {config['data_file_clicks']}")

    log.info("Creating sessions...")
    sessions = create_sessions(click_events, dataset_name, "clicks")
    log.info(f"Created {len(sessions)} sessions for {dataset_name}")

    log.info("Filtering sessions...")
    sessions = apply_session_filtering(sessions)
    log.info(f"Remaining sessions after filtering: {len(sessions)}")

    log.info("Splitting sessions into train and test...")
    train_sessions, test_sessions = train_test_split(sessions, dataset_name, config["split_seconds"], config["split_idx"])
    log.info(f"Split sessions into {len(train_sessions)} train and {len(test_sessions)} test sessions")

    train_aids = [event["aid"] for session in train_sessions for event in session["events"]]

    test_sessions = filter_aids(train_aids, test_sessions)

    test_sessions = filter_short_sessions(test_sessions)
    log.info(f"Remaining test sessions after filtering: {len(test_sessions)}")

    log.info("Creating item indices...")
    aid_to_idx = create_aid_to_idx(train_aids)
    log.info(f"Created {len(aid_to_idx)} item indices")

    order_events = read_file(config["data_file_orders"], header=config["header"])

    log.info(f"Read {len(order_events)} events from {config['data_file_orders']}")

    order_sessions = create_sessions(order_events, dataset_name, "orders")

    log.info(f"Created {len(order_sessions)} order sessions for {dataset_name}")

    log.info(f"Keep train aids in orders")
    order_sessions = filter_aids(train_aids, order_sessions)

    log.info(f"Merge orders to train sessions")
    train_sessions = merge_clicks_and_orders(train_sessions, order_sessions)

    log.info(f"Merge orders to test sessions")
    test_sessions = merge_clicks_and_orders(test_sessions, order_sessions)

    log.info("Remapping item indices...")
    train_sessions, num_train_sessions, num_train_click_events, num_train_order_events = remap_indices(
        train_sessions, aid_to_idx)
    test_sessions, num_test_sessions, num_test_click_events, num_test_order_events = remap_indices(test_sessions, aid_to_idx)

    log.info("Sorting sessions")
    train_sessions = sort_sessions(train_sessions)
    test_sessions = sort_sessions(test_sessions)

    output_dir = data_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing sessions to {output_dir}")
    write_file(train_sessions, output_dir / f"{dataset_name}_train.jsonl")
    write_file(test_sessions, output_dir / f"{dataset_name}_test.jsonl")

    stats_file = output_dir / f"{dataset_name}_stats.json"
    log.info(f"Writing stats to {stats_file}")
    write_stats(len(set(train_aids)), num_train_sessions, num_train_click_events, num_train_order_events, num_test_sessions,
                num_test_click_events, num_test_order_events, stats_file)


def filter_carts(in_file, out_file):
    # Change this function somehow
    num_sessions = 0
    num_click_events = 0
    num_order_events = 0

    items = set()
    log.info(f"Filtering carts from {in_file} to {out_file}")
    with open(in_file, "r") as read_file:
        with open(out_file, "w") as write_file:
            for line in read_file:
                session = json.loads(line)
                session["events"] = list(filter(lambda d: d['type'] != "carts", session["events"]))
                session["events"] = increment_aids(session["events"])
                num_sessions += 1
                num_click_events += len(list(filter(lambda d: d['type'] == "clicks", session["events"])))
                num_order_events += len(list(filter(lambda d: d['type'] == "orders", session["events"])))
                items.update([event["aid"] for event in session["events"] if event["type"] == "clicks"])
                write_file.write(json.dumps(session, separators=(',', ':')) + "\n")
                if num_sessions % 1_000_000 == 0:
                    log.info(f"Processed {num_sessions} sessions")
    return num_sessions, num_click_events, num_order_events, len(items)


def increment_aids(events):
    for event in events:
        event["aid"] = event["aid"] + 1
    return events


def run_preprocessing_otto(data_dir):
    # change function filter_non_clicks
    num_train_sessions, num_train_click_events, num_train_order_events, num_items = filter_carts(
        f"{data_dir}/otto/otto-recsys-train.jsonl", f"{data_dir}/otto/otto_train.jsonl")
    num_test_sessions, num_test_click_events, num_test_order_events, _ = filter_carts(f"{data_dir}/otto/otto-recsys-test.jsonl",
                                                                                      f"{data_dir}/otto/otto_test.jsonl")
    stats_file = f"{data_dir}/otto/otto_stats.json"
    log.info(f"Writing stats to {stats_file}")
    write_stats(num_items, num_train_sessions, num_train_click_events, num_train_order_events, num_test_sessions,
                num_test_click_events, num_test_order_events, stats_file)


class DatasetConf(Enum):
    YOOCHOOSE = 'yoochoose'
    DIGINETICA = 'diginetica'
    OTTO = 'otto'
    ALL = 'all'

    def __str__(self):
        return self.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=DatasetConf, default=DatasetConf.ALL)
    parser.add_argument("--data_dir", type=str, default="datasets")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    log.basicConfig(level=log.INFO)
    log.info(f"Running preprocessing for {args.dataset} dataset")

    yoochoose_conf = {
        "dataset_name": "yoochoose",
        "data_file_clicks": data_dir / "yoochoose" / "yoochoose-clicks.dat",
        "data_file_orders": data_dir / "yoochoose" / "yoochoose-buys.dat",
        "header": False,
        "split_seconds": 86400 * 1,  # 1 day (for testing)
        "split_idx": -1  # use last session timestamp for split
    }

    diginetica_conf = {
        "dataset_name": "diginetica",
        "data_file_clicks": data_dir / "diginetica" / "train-item-views.csv",
        "data_file_orders": data_dir / "diginetica" / "train-purchases.csv",
        "header": True,
        "split_seconds": 86400 * 7,  # 7 days (for testing)
        "split_idx": 0  # use first session timestamp for split
    }

    if args.dataset == DatasetConf.YOOCHOOSE:
        run_preprocessing(yoochoose_conf, data_dir)
    elif args.dataset == DatasetConf.DIGINETICA:
        run_preprocessing(diginetica_conf, data_dir)
    elif args.dataset == DatasetConf.OTTO:
        run_preprocessing_otto(data_dir)
    elif args.dataset == DatasetConf.ALL:
        run_preprocessing(yoochoose_conf, data_dir)
        run_preprocessing(diginetica_conf, data_dir)
        run_preprocessing_otto(data_dir)

    log.info("All done!")


if __name__ == "__main__":
    main()
