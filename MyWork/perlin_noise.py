import numpy as np
import torch


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def Perlin_Noise_Creator(dim, color):
    '''
    Input: dim (int), dimension of the final image
           color (tensor), maximum color of the Perlin Noise
    Outpput: Perlin Noise tensor
    '''

    # Creation of the noise (np-array) using the perlin_noise.py
    noise = generate_perlin_noise_2d((dim,dim), (8, 8))

    # Scale and translation. From (-1,1) to (0,1)
    noise = noise + 1
    noise = noise/2

    # Creation of the Perlin Noise tensor
    pn_tensor = torch.from_numpy(noise) #(dim,dim)
    pn_tensor_3dim = pn_tensor.unsqueeze(0) #(1,dim,dim)

    # Use the color to change the Perlin Noise
    tensor_channel1 = pn_tensor_3dim.clone()*color[0]
    tensor_channel2 = pn_tensor_3dim.clone()*color[1]
    tensor_channel3 = pn_tensor_3dim.clone()*color[2]
    total = torch.stack([tensor_channel1, tensor_channel2, tensor_channel3], dim=1)
    
    final_tensor = total.squeeze(0)

    return final_tensor

def Inverted_Perlin_Noise_Creator(dim, color):
    '''
    Input: dim (int), dimension of the final image
           color (tensor), minimum color of the Perlin Noise
    Outpput: Perlin Noise tensor
    '''

    # Create the color
    color_image = color.unsqueeze(-1).unsqueeze(-1)
    color_image = color_image.expand(-1, dim, dim)

    # Creation of the noise (np-array) using the perlin_noise.py
    noise = generate_perlin_noise_2d((dim,dim), (8, 8))

    # Scale and translation. From (-1,1) to (0,1)
    noise = noise + 1
    noise = noise/2

    # Creation of the Perlin Noise tensor
    pn_tensor = torch.from_numpy(noise) #(dim,dim)
    pn_tensor_3dim = pn_tensor.unsqueeze(0) #(1,dim,dim)

    # Use the color to change the Perlin Noise
    tensor_channel1 = pn_tensor_3dim.clone()*(1-color[0])
    tensor_channel2 = pn_tensor_3dim.clone()*(1-color[1])
    tensor_channel3 = pn_tensor_3dim.clone()*(1-color[2])
    total = torch.stack([tensor_channel1, tensor_channel2, tensor_channel3], dim=1)
    
    final_tensor = color_image + total.squeeze(0)

    return final_tensor

def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.
    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise