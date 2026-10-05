from PIL import Image
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

map_dir = Path(__file__).parent.parent
map_path = map_dir / "maps" / "gauntlet.pgm"
Image.open(map_path).show()
res, sigma = 0.05, 0.20                       
m   = np.array(Image.open(map_path))
occ = (m < 89).astype(float)                   
F   = gaussian_filter(occ, sigma / res)        
F  /= F.max()                                
img = Image.fromarray(((1 - F) * 255).astype(np.uint8))
img.resize((img.width * 8, img.height * 8), Image.NEAREST).show()
