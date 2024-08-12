import gc
import json
from urllib.parse import parse_qs
from io import BytesIO

import torch
import soundfile as sf

from commons import intersperse
from symbols import pad, waitau_symbols, hakka_symbols, waitau_symbol_to_id, hakka_symbol_to_id
from utils import load_model

class VoiceError(Exception): pass
class ToneError(Exception): pass
class SymbolError(Exception): pass

def application(environ, start_response):
    code, content = app(environ.get("PATH_INFO"), environ.get("QUERY_STRING"))
    if code == 200:
        mime = "audio/mpeg"
    else:
        mime = "application/json"
        content = json.dumps(content).encode()
    start_response(str(code), [
        ("Content-Type", mime),
        ("Content-Length", str(len(content))),
        ("Access-Control-Allow-Origin", "*"),
        ("Cache-Control", "max-age=604800, public"),
        ("Connection", "Keep-Alive"),
        ("Keep-Alive", "timeout=120"),
    ])
    yield content

def app(path, query):
    try:
        function, language, text = path.encode("latin_1").decode().lower().strip("/").split("/")
        if function != "tts" or language not in {"waitau", "hakka"}: raise ValueError("Invalid URI segment")
        voice = parse_qs(query.encode("latin_1").decode().lower(),
                         keep_blank_values=True, strict_parsing=True, errors="strict").get("voice", ["male"])
        if len(voice) != 1: raise VoiceError(f"Expected at most a single 'voice', received {len(voice)} voices")
        voice = voice[0]
        if voice not in {"male", "female"}: raise VoiceError(f"The 'voice' query must be either 'male' or 'female' (defaults to 'male'), received '{voice}'")
    except UnicodeError as err:
        codec, content, start, end, reason = err.args
        content = content[start:end]
        if isinstance(content, bytes): content = content.decode("latin_1")
        return (500, {"error": "Error while decoding URI: invalid characters", "message": content})
    except VoiceError as err:
        return (500, {"error": "Invalid voice", "message": str(err)})
    except ValueError:
        return (404, {"error": "Page not found"})
    try:
        buffer = BytesIO()
        sf.write(buffer, generate_audio(language, voice, text.replace("+", " ")), 44100, format="WAV")
        return (200, buffer.getvalue())
    except ToneError as err:
        return (500, {"error": "Invalid syllable", "message": str(err)})
    except SymbolError as err:
        return (500, {"error": "Unrecognized symbol", "message": str(err.__cause__)})
    except Exception as err:
        return (500, {"error": "Unexpected error", "message": type(err).__name__ + ": " + str(err)})

models = {}
device = "cpu"

def generate_audio(language, voice, text):
    global models

    name = f"{language}_{voice}"
    if name not in models:
        models[name] = load_model(f"data/{name}.pth", "data/config.json", len(waitau_symbols if language == "waitau" else hakka_symbols))

    phones, tones, word2ph = [pad], [0], [1]
    for syllable in text.split():
        if len(syllable) == 1:
            phones.append(syllable)
            tones.append(0)
            word2ph.append(1)
            continue
        try:
            tone = int(syllable[-1], base=7)
        except ValueError as err:
            raise ToneError(f"'{syllable}' does not end with tone 0~6") from err
        it = (i for i, c in enumerate(syllable) if c in "aeiouäöüæ")
        index = next(it, 0)
        initial = syllable[:index]
        if language == "waitau":
            final = syllable[index:-1]
            phones += [initial, final]
            tones += [tone, tone]
            word2ph.append(2)
        else:
            medial = "i" if initial == "y" else "#"
            final_index = index
            if syllable[index] == "i":
                final_index = next(it, index)
                if final_index != index:
                    medial = "i"
            final = syllable[final_index:-1]
            phones += [initial, medial, final]
            tones += [tone, 0 if medial == "#" else tone, tone]
            word2ph.append(3)

    phones.append(pad)
    tones.append(0)
    word2ph.append(1)
    symbol_to_id = waitau_symbol_to_id if language == "waitau" else hakka_symbol_to_id
    try:
        phones = [symbol_to_id[symbol] for symbol in phones]
    except KeyError as err:
        raise SymbolError() from err
    lang_ids = [0] * len(phones)

    phones = intersperse(phones, 0)
    tones = intersperse(tones, 0)
    lang_ids = intersperse(lang_ids, 0)
    word2ph = [n * 2 for n in word2ph]
    word2ph[0] += 1
    del word2ph

    phones = torch.LongTensor(phones)
    tones = torch.LongTensor(tones)
    lang_ids = torch.LongTensor(lang_ids)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([0]).to(device)
        audio = (
            models[name].infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del (
            x_tst,
            tones,
            lang_ids,
            x_tst_lengths,
            speakers,
        )
        gc.collect()
    return audio
