"""
Service FFmpeg pour concaténation audio - Version Coolify
"""

import os
import asyncio
import uuid
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="FFmpeg Audio Service", version="1.0.0")

# Configuration
TEMP_DIR = Path("/tmp/audio")
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB par fichier
MAX_FILES = 200
CLEANUP_AFTER_HOURS = 1


class ConcatRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, max_length=MAX_FILES)
    output_format: str = Field(default="mp3")
    silence_between: float = Field(default=0.3, ge=0, le=5.0)
    normalize: bool = Field(default=True)


class AudioInfo(BaseModel):
    duration_seconds: float
    file_size_bytes: int
    format: str
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


async def download_file(session: aiohttp.ClientSession, url: str, dest_path: Path) -> bool:
    """Télécharge un fichier depuis une URL."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
            if response.status != 200:
                print(f"Erreur téléchargement {url}: status {response.status}")
                return False
            
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                print(f"Fichier trop gros: {url}")
                return False
            
            async with aiofiles.open(dest_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
            
            return True
    except Exception as e:
        print(f"Erreur téléchargement {url}: {e}")
        return False


def get_audio_duration(file_path: Path) -> float:
    """Obtient la durée d'un fichier audio."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def create_silence(duration: float, output_path: Path, sample_rate: int = 44100) -> bool:
    """Crée un fichier de silence."""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'anullsrc=r={sample_rate}:cl=stereo',
            '-t', str(duration),
            '-acodec', 'libmp3lame', '-q:a', '2',
            str(output_path)
        ], capture_output=True, timeout=30, check=True)
        return True
    except:
        return False


def concatenate_audio_files(file_list_path: Path, output_path: Path, normalize: bool = True) -> bool:
    """Concatène les fichiers audio listés."""
    try:
        if normalize:
            # Avec normalisation loudnorm
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(file_list_path),
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
                '-acodec', 'libmp3lame', '-q:a', '2',
                str(output_path)
            ]
        else:
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(file_list_path),
                '-acodec', 'libmp3lame', '-q:a', '2',
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        return result.returncode == 0
    except Exception as e:
        print(f"Erreur concaténation: {e}")
        return False


async def cleanup_old_files():
    """Nettoie les fichiers plus vieux que CLEANUP_AFTER_HOURS."""
    try:
        cutoff = datetime.now() - timedelta(hours=CLEANUP_AFTER_HOURS)
        for item in TEMP_DIR.iterdir():
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    item.unlink()
            elif item.is_dir():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    for f in item.iterdir():
                        f.unlink()
                    item.rmdir()
    except Exception as e:
        print(f"Erreur nettoyage: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Vérifier que FFmpeg est disponible
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        ffmpeg_ok = result.returncode == 0
    except:
        ffmpeg_ok = False
    
    return {
        "status": "healthy" if ffmpeg_ok else "unhealthy",
        "ffmpeg_available": ffmpeg_ok,
        "temp_dir": str(TEMP_DIR),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/concat")
async def concat_audio(request: ConcatRequest, background_tasks: BackgroundTasks):
    """Concatène plusieurs fichiers audio en un seul."""
    
    job_id = str(uuid.uuid4())[:8]
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        # Télécharger tous les fichiers
        downloaded_files = []
        async with aiohttp.ClientSession() as session:
            for i, url in enumerate(request.urls):
                file_path = job_dir / f"segment_{i:04d}.mp3"
                success = await download_file(session, url, file_path)
                if success and file_path.exists():
                    downloaded_files.append(file_path)
                else:
                    print(f"Échec téléchargement segment {i}: {url}")
        
        if len(downloaded_files) == 0:
            raise HTTPException(status_code=400, detail="Aucun fichier n'a pu être téléchargé")
        
        if len(downloaded_files) != len(request.urls):
            print(f"Attention: {len(downloaded_files)}/{len(request.urls)} fichiers téléchargés")
        
        # Créer le fichier de silence si nécessaire
        silence_file = None
        if request.silence_between > 0:
            silence_file = job_dir / "silence.mp3"
            create_silence(request.silence_between, silence_file)
        
        # Créer la liste des fichiers pour FFmpeg
        file_list_path = job_dir / "files.txt"
        with open(file_list_path, 'w') as f:
            for i, audio_file in enumerate(downloaded_files):
                f.write(f"file '{audio_file}'\n")
                if silence_file and silence_file.exists() and i < len(downloaded_files) - 1:
                    f.write(f"file '{silence_file}'\n")
        
        # Concaténer
        output_file = job_dir / f"output.{request.output_format}"
        success = concatenate_audio_files(file_list_path, output_file, request.normalize)
        
        if not success or not output_file.exists():
            raise HTTPException(status_code=500, detail="Échec de la concaténation audio")
        
        # Obtenir les infos du fichier final
        duration = get_audio_duration(output_file)
        file_size = output_file.stat().st_size
        
        # Planifier le nettoyage
        background_tasks.add_task(cleanup_old_files)
        
        return {
            "success": True,
            "job_id": job_id,
            "download_url": f"/download/{job_id}/output.{request.output_format}",
            "duration_seconds": round(duration, 2),
            "file_size_bytes": file_size,
            "segments_processed": len(downloaded_files),
            "segments_requested": len(request.urls)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{job_id}/{filename}")
async def download_file_endpoint(job_id: str, filename: str):
    """Télécharge un fichier traité."""
    file_path = TEMP_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/mpeg"
    )


@app.get("/info/{job_id}/{filename}")
async def get_file_info(job_id: str, filename: str):
    """Obtient les infos d'un fichier audio."""
    file_path = TEMP_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    duration = get_audio_duration(file_path)
    file_size = file_path.stat().st_size
    
    return {
        "duration_seconds": round(duration, 2),
        "file_size_bytes": file_size,
        "filename": filename
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
