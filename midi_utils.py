import mido


def convert_midi_event(mid_file):
    """mid_file (str): path to midi file"""
    mid = mido.MidiFile(mid_file)
    current_time = 0
    out_list = []
    for msg in mid:
        current_time += msg.time
        # For note off the type is still "note_on" but with velocity=0
        if msg.type == "note_on":
            out_list.append(
                {"time": current_time, "note": msg.note, "vel": msg.velocity}
            )
    return out_list


def convert_for_scoring(midi_events):
    """Create list for pitch, interval and velocity"""
    ref_pitches = []
    ref_velocity = []
    ref_interval = []
    on_dict = {}
    for event in midi_events:
        if event["vel"] > 0:
            # To handle no off event for note
            if event["note"] in on_dict:
                ref_interval[on_dict[event["note"]]] = [
                    ref_interval[on_dict[event["note"]]][0],
                    event["time"],
                ]
            ref_pitches.append(event["note"])
            ref_velocity.append(event["vel"])
            ref_interval.append([event["time"], 0])
            on_dict[event["note"]] = len(ref_pitches) - 1
        elif event["note"] in on_dict:
            # print(ref_interval)
            # print(on_dict)
            # print(ref_pitches)
            ref_interval[on_dict[event["note"]]] = [
                ref_interval[on_dict[event["note"]]][0],
                event["time"],
            ]
            del on_dict[event["note"]]
    if on_dict:
        print(on_dict)
    return ref_pitches, ref_interval, ref_velocity


def get_suit_timing(midi_events):
    on_list = []
    suit_list = []
    event_str = ""
    start_time = 0
    for event in midi_events:
        if event["vel"] > 0:
            on_list.append(event["note"])
        else:
            if event["note"] not in on_list:
                print(event)
                print(midi_events)
            else:
                on_list.remove(event["note"])

        event_str += f" <vel:{event['vel']}> <note:{event['note']}>"

        if not on_list:
            suit_list.append(
                {"start": start_time, "end": event["time"], "stm": event_str}
            )
            event_str = ""
            start_time = event["time"]

    return suit_list
