"""Un solo comando per preparare i dati e addestrare il modello BBCH."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Facoltativo: configurazione diversa.")
    args = parser.parse_args()
    try:
        import copy
        import os
        import shutil
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        import torch
        from modello.supporto import impostazioni, project_path, read_json, write_json, file_hash
        from modello.dati import prepara_dati
        from modello.addestramento import run_training

        settings = impostazioni(args.config)
        advanced = read_json(project_path("configs/model.json"))
        repo = project_path(settings["ambiente"]["dinov2"])
        if not (repo / "hubconf.py").is_file():
            raise ValueError("Non trovo la copia locale ufficiale DINOv2. Controlla ambiente.dinov2 in config.json.")
        if not (Path(torch.hub.get_dir()) / "checkpoints/dinov2_vitb14_pretrain.pth").is_file():
            raise ValueError("Mancano i pesi ufficiali DINOv2 nella cache Torch. Chiedi al responsabile tecnico di predisporli.")
        user = settings["addestramento"]
        if min(user["epoche_prima_fase"], user["epoche_seconda_fase"], user["immagini_insieme"], user["immagini_per_aggiornamento"]) < 1:
            raise ValueError("Epoche e quantità di immagini devono essere interi positivi.")
        if user["immagini_per_aggiornamento"] % user["immagini_insieme"]:
            raise ValueError("'immagini_per_aggiornamento' deve essere multiplo di 'immagini_insieme'.")
        if user["lato_immagini"] < 14 or user["lato_immagini"] % 14:
            raise ValueError("'lato_immagini' deve essere un multiplo di 14, per esempio 336.")
        if user["processi_lettura"] < 0:
            raise ValueError("'processi_lettura' non può essere negativo.")
        output = project_path(settings["risultati"])
        runs = output / "addestramenti"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC-")
        run = Path(tempfile.mkdtemp(prefix=stamp, dir=runs))
        print("Controllo le etichette e preparo i dati. Le immagini originali non saranno modificate.", flush=True)
        splits = prepara_dati(settings, run)
        training = copy.deepcopy(advanced["training_defaults"])
        training.update(model_config="configs/model.json", image_root=str(project_path(settings["dati"]["immagini"])),
            dinov2_repo=str(repo), splits_dir=str(splits), output_dir=str(run / "modello"),
            device=settings["ambiente"]["dispositivo"], seed=settings["suddivisione"]["seme"],
            workers=user["processi_lettura"], input_size=user["lato_immagini"], cache_batch_size=user["immagini_insieme"])
        training["stage1"]["epochs"] = user["epoche_prima_fase"]
        training["stage2"].update(epochs=user["epoche_seconda_fase"], physical_batch_size=user["immagini_insieme"],
            effective_batch_size=user["immagini_per_aggiornamento"])
        write_json(run / "impostazioni.json", settings)
        print("Avvio dai pesi ufficiali DINOv2, non dal precedente modello BBCH.", flush=True)
        print("F1 è un indice di riconoscimento: qui do lo stesso peso a ogni classe. Il test finale resta chiuso.", flush=True)
        checkpoint = run_training(training)
        fd, temporary = tempfile.mkstemp(prefix=".modello-", suffix=".pt", dir=output)
        os.close(fd)
        try:
            shutil.copyfile(checkpoint, temporary)
            os.replace(temporary, output / "modello_corrente.pt")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        write_json(output / "modello_corrente.json", {"checkpoint_versionato": str(checkpoint.relative_to(output)),
            "sha256": file_hash(checkpoint), "test_finale_eseguito": False})
        print(f"Addestramento terminato. Modello pronto per predict.py: {output / 'modello_corrente.pt'}", flush=True)
        print(f"Log e versione conservati in: {run}", flush=True)
        return 0
    except ModuleNotFoundError as error:
        print(f"Manca una dipendenza ({error.name}). Attiva .venv e usa: python -m pip install -r requirements.txt", file=sys.stderr)
    except KeyboardInterrupt:
        print("Operazione interrotta. Il modello precedente non viene sostituito. Rilancia train.py per ricominciare.", file=sys.stderr)
        return 130
    except Exception as error:
        from modello.supporto import messaggio_errore
        print(f"Addestramento non completato: {messaggio_errore(error)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
