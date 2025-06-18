import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from MicroscopicModel import plotrod, plotAnts
import subprocess
from tqdm import tqdm
import multiprocessing as mp

def animate_from_npz_funcAnimation(npz_file="soft_pendulum_microscopic.npz",
                                   out_file="rod_ants",
                                   rod_length=5.0,
                                   skipEvery=200,
                                   output_format="gif"):

    data = np.load(npz_file, allow_pickle=True)
    time_array   = data["time"][::skipEvery]
    q_array      = data["q"][::skipEvery]
    ants_array   = data["ants"][::skipEvery]
    angles_array = data["angles"][::skipEvery]

    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame_idx):
        ax.clear()

        t   = time_array[frame_idx]
        q   = q_array[frame_idx]
        ants= ants_array[frame_idx]
        angs= angles_array[frame_idx]

        class FakeAnts:
            pass
        FA = FakeAnts()
        FA.ants   = ants
        FA.angles = angs
        FA.nv     = ants.shape[0]

        plotrod(q, t, rod_length)

        plotAnts(q, FA)

        ax.set_aspect('equal', 'box')
        ax.set_xlim([-1.2*rod_length, 1.2*rod_length])
        ax.set_ylim([-1.2*rod_length, 1.2*rod_length])
        ax.set_title(f"L = {rod_length} cm, Time: {t:.2f} s")
        

        pbar.update(1)

        return []

    nFrames = len(time_array)
    ani = FuncAnimation(
        fig,
        update,
        frames=range(nFrames),
        interval=50,
        blit=False
    )

    if output_format == "gif":
        writer = PillowWriter(fps=20)
    elif output_format == "mp4":
        writer = FFMpegWriter(fps=20)
    else:
        raise ValueError("Unsupported output format. Use 'gif' or 'mp4'.")

    outfile = out_file + "." + output_format

    print(f"Saving animation to {outfile} using {writer}")
    pbar = tqdm(total=len(time_array), desc="Creating animation")
    try:
        ani.save(outfile, writer=writer)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with error: {e}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.output}")
    plt.close(fig)
    pbar.close()
    print(f"Animation saved as {outfile}")

relpath=r'sim_20250618\rod_L20.0cm_seg0.5cm.npz'
def run_animation(relpath=relpath):
        tokens = relpath.split("_")
        for token in tokens:
            if "L" in token:
                L = float(token.split("L")[1].split("cm")[0])
                print(f"Rod length found: {L} cm")
        
        outfile = relpath.rstrip(".npz")
        outfile += "_animation"
        animate_from_npz_funcAnimation(npz_file=relpath,
                                       out_file=outfile,
                                       rod_length=L,
                                       skipEvery=10,
                                       output_format="mp4")
        
if __name__ == "__main__":

    p = mp.Process(target=run_animation)
    p.start()
    p.join()
