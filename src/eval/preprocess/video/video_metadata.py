import ffmpeg

def get_video_metadata(video_path):
    try:
        # Probe the video file for structural metadata
        probe = ffmpeg.probe(video_path)
        
        # Isolate format (container) and stream (codec) properties
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        metadata = {
            "format": probe['format'].get('format_long_name'),
            "duration_seconds": float(probe['format'].get('duration', 0)),
            "size_bytes": int(probe['format'].get('size', 0)),
            "bit_rate": int(probe['format'].get('bit_rate', 0)),
        }
        
        if video_stream:
            metadata.update({
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "codec": video_stream.get('codec_name'),
                "frame_rate": video_stream.get('avg_frame_rate')
            })
            
        return metadata

    except ffmpeg.Error as e:
        print(f"Error occurred: {e.stderr.decode()}")
        return None
