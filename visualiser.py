import argparse
import re
import sys
import threading
import time
from pathlib import Path

import pygame
import sherpa_onnx
import sounddevice as sd


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), f"{filename} does not exist!\n"


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=4,
        help="""Used only when --decoding-method is modified_beam_search.
        It specifies number of active paths to keep during decoding.
        """,
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    return parser.parse_args()


def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
    )
    return recognizer


args = get_args()

devices = sd.query_devices()
if len(devices) == 0:
    print("No microphone devices found")
    sys.exit(0)

# print(devices)
# sd.default.device=0
default_input_device_idx = sd.default.device[0]
print(f'Use default device: {devices[default_input_device_idx]["name"]}')

recognizer = create_recognizer(args)

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 780, 500
BACKGROUND_COLOR = (30, 30, 30)
FPS = 60
ACTIVE_WHITE_KEY_COLOR = (255, 200, 200)  # Highlighted color for active white keys
ACTIVE_BLACK_KEY_COLOR = (200, 0, 0)  # Highlighted color for active black keys
KEYBOARD_HEIGHT = 100  # Height of the piano keyboard at the bottom
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Real-Time MIDI Note Visualization")

# MIDI note visualization settings
MIDI_MIN, MIDI_MAX = 21, 108  # Piano range (MIDI note numbers)
# NOTE_COLOR = (100, 250, 100)
NOTE_HEIGHT = 30  # Width of each note rectangle
NOTE_SCROLL_SPEED = 2  # Pixels per frame

# Define colors
WHITE_KEY_COLOR = (255, 255, 255)
BLACK_KEY_COLOR = (0, 0, 0)
ACTIVE_KEY_COLOR = (255, 100, 100)
# Key dimensions
WHITE_KEY_WIDTH = SCREEN_WIDTH // 52
BLACK_KEY_WIDTH = WHITE_KEY_WIDTH // 2
BLACK_KEY_HEIGHT = KEYBOARD_HEIGHT // 1.5

# Function to map MIDI note to horizontal position
# def midi_to_x_position(midi_note):
#     return int((midi_note - MIDI_MIN) / (MIDI_MAX - MIDI_MIN) * SCREEN_WIDTH)
# Note-to-position mapping for white and black keys
WHITE_KEYS = [0, 2, 4, 5, 7, 9, 11]  # White keys in each octave
BLACK_KEYS = [1, 3, 6, 8, 10]  # Black keys in each octave

# Precompute a dictionary mapping each MIDI note to (note_in_octave, octave)
MIDI_NOTE_MAP = {}
white_key_x = 0
for midi_note in range(MIDI_MIN, MIDI_MAX + 1):
    note_in_octave = midi_note % 12
    octave = midi_note // 12 - 1
    octave_offset = ((octave - 1) * 7 * WHITE_KEY_WIDTH) + (
        2 * WHITE_KEY_WIDTH
    )  # Offset by octave
    is_white_key = note_in_octave not in BLACK_KEYS
    if is_white_key:
        MIDI_NOTE_MAP[midi_note] = (white_key_x, "white")
        white_key_x += WHITE_KEY_WIDTH
    else:
        black_key_x = white_key_x - BLACK_KEY_WIDTH // 2
        MIDI_NOTE_MAP[midi_note] = (black_key_x, "black")

    # if note_in_octave in WHITE_KEYS:
    #     # White key
    #     key_index = WHITE_KEYS.index(note_in_octave)
    #     x_position = octave_offset + key_index * WHITE_KEY_WIDTH
    #     MIDI_NOTE_MAP[midi_note] = (x_position, "white")
    # else:
    #     # Black key
    #     key_index = BLACK_KEYS.index(note_in_octave)
    #     x_position = octave_offset + (key_index + 1) * WHITE_KEY_WIDTH - BLACK_KEY_WIDTH // 2
    #     MIDI_NOTE_MAP[midi_note] = (x_position, "black")
    #     # width = BLACK_KEY_WIDTH


def midi_to_keyboard_position(midi_note):
    """Return x position and width of the note to match the keyboard key."""
    if midi_note not in MIDI_NOTE_MAP:
        raise ValueError("MIDI note is out of supported range")

    # Get precomputed values
    note_in_octave, octave = MIDI_NOTE_MAP[midi_note]
    # print(f"Note {note_in_octave} in octave {octave}")
    octave_offset = ((octave - 1) * 7 * WHITE_KEY_WIDTH) + (
        2 * WHITE_KEY_WIDTH
    )  # Offset by octave
    width = WHITE_KEY_WIDTH  # Default width for white keys
    x_position = 0

    # Determine if the note is white or black
    if note_in_octave in WHITE_KEYS:
        # White key
        key_index = WHITE_KEYS.index(note_in_octave)
        x_position = octave_offset + key_index * WHITE_KEY_WIDTH
    else:
        # Black key
        key_index = BLACK_KEYS.index(note_in_octave)
        x_position = (
            octave_offset + (key_index + 1) * WHITE_KEY_WIDTH - BLACK_KEY_WIDTH // 2
        )
        width = BLACK_KEY_WIDTH

    return x_position, width


# Function to calculate color based on MIDI note (red to blue)
def note_to_color(midi_note):
    # Normalize the MIDI note to a range from 0 to 1
    normalized_value = (midi_note - MIDI_MIN) / (MIDI_MAX - MIDI_MIN)
    # Calculate red and blue values based on normalized_value
    red = int((1 - normalized_value) * 255)  # Lower notes are redder
    blue = int(normalized_value * 255)  # Higher notes are bluer
    return (red, 0, blue)


class MidiNote:
    def __init__(self, midi_note):
        self.midi_note = midi_note
        self.color = note_to_color(midi_note)  # Set the color based on the note
        self.x_pos, color = MIDI_NOTE_MAP[midi_note]
        self.width = WHITE_KEY_WIDTH if color == "white" else BLACK_KEY_WIDTH
        self.y_pos = (
            SCREEN_HEIGHT - KEYBOARD_HEIGHT - NOTE_HEIGHT
        )  # Start above the keyboard
        self.rect = pygame.Rect(self.x_pos, self.y_pos, self.width, NOTE_HEIGHT)

    def update(self):
        # Move the note upwards to create a bottom-to-top scrolling effect
        self.y_pos -= NOTE_SCROLL_SPEED
        self.rect.y = self.y_pos

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)


# Draw the piano keyboard
def draw_piano_keyboard(active_notes):
    # white_key_x = 0
    # black_key_offsets = [0, 1, 3, 4, 5]  # Black keys relative to white keys in each octave

    # for midi_note in range(MIDI_MIN, MIDI_MAX + 1):
    #     is_white_key = (midi_note % 12) not in [1, 3, 6, 8, 10]
    #     is_active = midi_note in active_notes
    for midi_note in range(MIDI_MIN, MIDI_MAX + 1):
        x_position, color = MIDI_NOTE_MAP[midi_note]
        is_white_key = True if color == "white" else False
        # is_white_key = note_in_octave not in BLACK_KEYS
        is_active = midi_note in active_notes
        #         x_position, width = midi_to_keyboard_position(midi_note)

        if is_white_key:
            key_rect = pygame.Rect(
                x_position,
                SCREEN_HEIGHT - KEYBOARD_HEIGHT,
                WHITE_KEY_WIDTH,
                KEYBOARD_HEIGHT,
            )
            color = ACTIVE_KEY_COLOR if is_active else WHITE_KEY_COLOR
            pygame.draw.rect(screen, color, key_rect)
            pygame.draw.rect(screen, BLACK_KEY_COLOR, key_rect, 1)
            # white_key_x += WHITE_KEY_WIDTH
        else:
            # black_key_x = white_key_x - BLACK_KEY_WIDTH // 2
            key_rect = pygame.Rect(
                x_position,
                SCREEN_HEIGHT - KEYBOARD_HEIGHT,
                BLACK_KEY_WIDTH,
                BLACK_KEY_HEIGHT,
            )
            color = ACTIVE_KEY_COLOR if is_active else BLACK_KEY_COLOR
            pygame.draw.rect(screen, color, key_rect)


# def draw_piano_keyboard(screen, active_notes):
#     """Draws a piano keyboard at the bottom of the screen with active notes highlighted.

#     Args:
#         screen: The pygame surface to draw on.
#         active_notes: A list of MIDI notes that are currently active.
#     """
#     for midi_note in range(MIDI_MIN, MIDI_MAX + 1):
#         note_in_octave, octave = MIDI_NOTE_MAP[midi_note]
#         x_position, width = midi_to_keyboard_position(midi_note)

#         # Determine if the note is active and set the color accordingly
#         if midi_note in active_notes:
#             # Active notes use highlighted colors
#             if note_in_octave in WHITE_KEYS:
#                 key_color = ACTIVE_WHITE_KEY_COLOR
#                 key_height = 120  # Height for active white keys
#             else:
#                 key_color = ACTIVE_BLACK_KEY_COLOR
#                 key_height = 80   # Height for active black keys
#         else:
#             # Inactive notes use default colors
#             if note_in_octave in WHITE_KEYS:
#                 key_color = WHITE_KEY_COLOR
#                 key_height = 120  # Height for white keys
#             else:
#                 key_color = BLACK_KEY_COLOR
#                 key_height = 80   # Height for black keys

#         # Draw the key with the calculated position, color, and size
#         pygame.draw.rect(screen, key_color, (x_position, KEYBOARD_HEIGHT, width, key_height))


# Simulate real-time MIDI note streaming as a generator (replace with real data source)
def recognize_stream(recognizer):
    sample_rate = 16000
    samples_per_read = int(0.02 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    last_result_index = 0
    stream = recognizer.create_stream()
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)
            if last_result != result:
                last_result = result
                new_results = result[last_result_index:]
                last_result_index = len(result)
                # print(type(result))
                print(new_results, flush=True)
                yield new_results
        # while True:
        # yield random.randint(60, 80)  # Simulated MIDI note
        # yield 60
        # time.sleep(0.5)  # Simulate delay between notes


# Run the visualization
def run_visualization():
    clock = pygame.time.Clock()
    notes = []
    active_notes = set()

    # Thread to add new notes as they come from the generator
    def note_listener():
        for result in recognize_stream(recognizer):
            # Extract note numbers from the result string using regex
            note_numbers = re.findall(r"<note:(\d+)>", result)
            # Convert each note number to an integer and create a MidiNote
            for note_str in note_numbers:
                midi_note = int(note_str)
                # midi_note = int(result) # DEBUG
                active_notes.add(midi_note)
                notes.append(MidiNote(midi_note))
            time.sleep(
                0.02
            )  # Clear active notes after a brief moment for this simulation
            active_notes.clear()

    # Start the listener thread
    threading.Thread(target=note_listener, daemon=True).start()

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update note positions and remove notes that scroll off the screen
        for note in notes[:]:
            note.update()
            if note.y_pos < -NOTE_HEIGHT:  # Remove notes off screen
                notes.remove(note)

        # Drawing
        screen.fill(BACKGROUND_COLOR)
        draw_piano_keyboard(active_notes)
        for note in notes:
            note.draw(screen)

        pygame.display.flip()  # Update the display
        clock.tick(FPS)  # Limit to 60 FPS


run_visualization()
pygame.quit()
