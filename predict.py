"""Prevede il BBCH da un file o da una cartella usando il modello corrente."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("immagini", nargs="?", help="Facoltativo: file o cartella; altrimenti uso config.json.")
    parser.add_argument("--config", default="config.json", help="Facoltativo: configurazione diversa.")
    args = parser.parse_args()
    try:
        import csv
        import os
        import tempfile
        import torch
        from modello.supporto import impostazioni, project_path, choose_device
        from modello.architettura import load_checkpoint
        from modello.dati import image_tensor, make_evaluation_transform

        settings = impostazioni(args.config)
        source = project_path(args.immagini or settings["previsione"]["immagini"])
        extensions = {".jpg", ".jpeg", ".heic", ".heif", ".png"}
        if source.is_file():
            paths = [source] if source.suffix.lower() in extensions else []
        elif source.is_dir():
            paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        else:
            raise ValueError(f"Il percorso '{source}' non esiste. Metti lì le immagini oppure cambia previsione.immagini in config.json.")
        if not paths:
            raise ValueError(f"Nessuna immagine trovata in '{source}'. Usa JPEG/JPG, HEIC/HEIF o PNG e controlla il percorso.")
        output = project_path(settings["risultati"])
        checkpoint = output / "modello_corrente.pt"
        if not checkpoint.is_file():
            raise ValueError("Nessun modello addestrato disponibile. Esegui prima: python train.py")
        device = choose_device(settings["ambiente"]["dispositivo"])
        model, config = load_checkpoint(checkpoint, project_path(settings["ambiente"]["dinov2"]), device,
                                        project_path("configs/model.json"))
        transform = make_evaluation_transform(config)
        rows = []
        print(f"Analizzo {len(paths)} immagini. La probabilità è una stima del modello, non una garanzia.", flush=True)
        with torch.inference_mode():
            for path in paths:
                logits, bbch_tensor = model(image_tensor(path, transform).unsqueeze(0).to(device))
                probability = torch.softmax(logits, dim=1)[0].cpu()
                index = int(probability.argmax())
                item = config["classes"][index]
                # Preserve the regression head's raw estimate. Do not clip it to
                # the classifier interval: a wrong class could otherwise create a
                # huge and misleading BBCH error.
                bbch_estimate = float(bbch_tensor[0].cpu())
                confidence = float(probability[index])
                row = {"immagine": str(path.resolve()), "classe": item["name"], "BBCH_min": item["bbch_min"],
                    "BBCH_max": item["bbch_max"], "fase": item["level1"],
                    "BBCH_stimato_float": round(bbch_estimate, 3),
                    "BBCH_stimato_display": f"{bbch_estimate:.3f}",
                    "confidence_score": confidence, "confidence_percent": round(confidence * 100, 2)}
                row.update({f"probabilita_{c['name']}": float(probability[c["id"]]) for c in config["classes"]})
                rows.append(row)
                print(f"{path.name}: classe {item['name']} — BBCH stimato {bbch_estimate:.3f} — confidence {confidence:.1%}", flush=True)
        fd, temporary = tempfile.mkstemp(prefix=".previsioni-", suffix=".csv", dir=output)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, output / "previsioni.csv")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"Risultato completo salvato in: {output / 'previsioni.csv'}", flush=True)
        return 0
    except ModuleNotFoundError as error:
        print(f"Manca una dipendenza ({error.name}). Attiva .venv e usa: python -m pip install -r requirements.txt", file=sys.stderr)
    except KeyboardInterrupt:
        print("Previsione interrotta. Il CSV precedente non viene sostituito.", file=sys.stderr)
        return 130
    except Exception as error:
        from modello.supporto import messaggio_errore
        print(f"Previsione non completata: {messaggio_errore(error)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
