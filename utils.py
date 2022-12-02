from typing import Union
import numpy as np
import torch
from functools import singledispatch
from math import pi
from matplotlib import pyplot as plt



__all__ = [
    "fft2",
    "ifft2",
    "normalize",
    "propagate_HK",
    "fidelity",
    "loc_fidelity",
    "performance_loc_fidelity",
    "performance_efficiency",
    "performance_crosstalk",
    "complim",
    "complim_subplot2",
    "plot_in_GS",
]


@singledispatch
def fft2(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    2D discrete forward fourier transform

    For any 2D array `x`
    assert(np.allclose(x, ifft2(fft2(x))))
    assert(np.allclose(x, fft2(ifft2(x))))
    """
    raise NotImplementedError(
        f"Cannot fourier transform `x` for type: {type(x)}")

@fft2.register
def _(x: np.ndarray) -> np.ndarray:
    """
    2D discrete forward fourier transform
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-1, -2)), norm="ortho"), axes=(-1, -2))

@fft2.register
def fft2_torch(x: torch.Tensor) -> torch.Tensor:
    """
    2D discrete forward fourier transform
    (x: torch.Tensor) -> torch.Tensor
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-1, -2)), norm="ortho"), dim=(-1, -2))



@singledispatch
def ifft2(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    2D discrete inverse fourier transform

    For any 2D array `x`
    assert(np.allclose(x, ifft2(fft2(x))))
    assert(np.allclose(x, fft2(ifft2(x))))
    """
    raise NotImplementedError(f"Cannot Inverse fourier transform `x` for {type(x)}")

@ifft2.register
def _(x: np.ndarray) -> np.ndarray:
    """
    2D discrete inverse fourier transform
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-1, -2)), norm="ortho"), axes=(-1, -2))

@ifft2.register
def ifft2_torch(x: torch.Tensor) -> torch.Tensor:
    """
    2D discrete inverse fourier transform
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-1, -2)), norm="ortho"), dim=(-1, -2))



@singledispatch
def normalize(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot normalize for {type(x)}")

@normalize.register
def _(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x)

@normalize.register
def _(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)



@singledispatch
def propagate_HK(FieldIn: Union[np.ndarray, torch.Tensor], kz: Union[np.ndarray, torch.Tensor], distance: float = 0.0) -> Union[np.ndarray, torch.Tensor]:
    """
    Free-space propagation. Mulitplied by (np.imag(kz)==0) to get rid of the evanescent components 

    """
    raise NotImplementedError(
        f"Cannot process `FieldIn` type: {type(FieldIn)}")

@propagate_HK.register
def _(FieldIn: np.ndarray, kz: np.ndarray, distance: float = 0.0) -> np.ndarray:
    FieldIn_FT = fft2(FieldIn)
    FieldOut_FT = FieldIn_FT*np.exp(1j*kz*distance)*(np.imag(kz)==0)
    FieldOut = ifft2(FieldOut_FT)
    return FieldOut

@propagate_HK.register
def _(FieldIn: torch.Tensor, kz: torch.Tensor, distance: float = 0.0) -> torch.Tensor:
    FieldIn_FT = fft2(FieldIn)
    FieldOut_FT = FieldIn_FT*torch.exp(1j*kz*distance)*(torch.imag(kz)==0)
    FieldOut = ifft2(FieldOut_FT)
    return FieldOut




@singledispatch
def fidelity(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    # returns a float number between 0 and 1
    raise NotImplementedError(f"Cannot check fidelity of `a`, `b` for {type(a)}, {type(b)}")

@fidelity.register
def _(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(np.abs(np.sum(normalize(a).conj() * normalize(b))))

@fidelity.register
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.abs(torch.sum(normalize(a).conj() * normalize(b))))




@singledispatch
def loc_fidelity(a: Union[np.ndarray, torch.Tensor], channel: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    # returns a float number between 0 and 1
    raise NotImplementedError(f"Cannot check fidelity of `a`, `b` for {type(a)}, {type(b)}")

@loc_fidelity.register
def _(a: np.ndarray, channel: np.ndarray, b: np.ndarray) -> float:
    a = a*channel
    return np.square(np.abs(np.sum(normalize(a).conj() * normalize(b))))

@loc_fidelity.register
def _(a: torch.Tensor, channel: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a*channel
    return torch.square(torch.abs(torch.sum(normalize(a).conj() * normalize(b))))




@singledispatch
def performance_loc_fidelity(A: Union[np.ndarray, torch.Tensor], channels: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot check fidelity of `A`, `B` for {type(A)}, {type(B)}")

@performance_loc_fidelity.register
def _(A: np.ndarray, channels: np.ndarray, B: np.ndarray) -> Union[np.ndarray, float]:
    A = np.squeeze(A)
    B = np.squeeze(B)
    CH = np.squeeze(channels)
    fid_list = np.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        fid_list[i] = loc_fidelity(A[i,:,:], CH[i,:,:], B[i,:,:])
    av_loc_fid = 100*np.sum(fid_list)/A.shape[0]
    return av_loc_fid, fid_list

@performance_loc_fidelity.register
def _(A: torch.Tensor, channels: torch.Tensor, B: torch.Tensor) -> Union[torch.Tensor, float]:
    A = torch.squeeze(A)
    B = torch.squeeze(B)
    CH = torch.squeeze(channels)
    fid_list = torch.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        fid_list[i] = loc_fidelity(A[i,:,:], CH[i,:,:], B[i,:,:])
    av_loc_fid = 100*torch.sum(fid_list)/A.shape[0]
    return av_loc_fid, fid_list




@singledispatch
def performance_efficiency(A: Union[np.ndarray, torch.Tensor], channels: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot check efficiency of `A` for {type(A)}")

@performance_efficiency.register
def _(A: np.ndarray, channels: np.ndarray) -> Union[np.ndarray, float]:
    A = np.squeeze(A)
    CH = np.squeeze(channels)
    eff_list = np.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        eff_list[i] = np.sum(A[i,:,:]*CH[i,:,:])
    av_eff = 100*np.sum(eff_list)/A.shape[0]
    return av_eff, eff_list

@performance_efficiency.register
def _(A: torch.Tensor, channels: torch.Tensor) -> Union[torch.Tensor, float]:
    A = torch.squeeze(A)
    CH = torch.squeeze(channels)
    eff_list = torch.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        eff_list[i] = torch.sum(A[i,:,:]*CH[i,:,:])
    av_eff = 100*torch.sum(eff_list)/A.shape[0]
    return av_eff, eff_list



@singledispatch
def performance_crosstalk(A: Union[np.ndarray, torch.Tensor], channels: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot check cross-talk for {type(A)}")

@performance_crosstalk.register
def _(A: np.ndarray, channels: np.ndarray) -> Union[np.ndarray, np.ndarray, float]:
    A = np.squeeze(A)
    CH = np.squeeze(channels)
    crs_list = np.zeros((A.shape[0]))
    crs_matrix = np.zeros((A.shape[0],A.shape[0]))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            crs_matrix[i,j] = np.sum(A[j,:,:]*CH[i,:,:])
    for i in range(0, A.shape[0]): 
        crs_list[i] = 1 - (crs_matrix[i,i]/np.sum(crs_matrix[:,i]))
    av_crs = 100*np.sum(crs_list)/A.shape[0]
    return av_crs, crs_list, crs_matrix

@performance_crosstalk.register
def _(A: torch.Tensor, channels: torch.Tensor) -> Union[torch.Tensor, torch.Tensor, float]:
    A = torch.squeeze(A)
    CH = torch.squeeze(channels)
    crs_list = torch.zeros((A.shape[0]))
    crs_matrix = torch.zeros((A.shape[0],A.shape[0]))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            crs_matrix[i,j] = torch.sum(A[j,:,:]*CH[i,:,:])
    for i in range(0, A.shape[0]): 
        crs_list[i] = 1 - (crs_matrix[i,i]/torch.sum(crs_matrix[:,i]))
    av_crs = 100*torch.sum(crs_list)/A.shape[0]
    return av_crs, crs_list, crs_matrix







@singledispatch
def complim(x: Union[np.ndarray, torch.Tensor]):
    # visualize a complex field. brightness = amplitude, colour = phase
    raise NotImplementedError(
        f"Cannot visualize `x` for type: {type(x)}")

@complim.register
def _(x: np.ndarray) -> np.ndarray:
    mAx = np.amax(np.abs(x))
    M = x/mAx
    A = np.abs(M)
    P = np.angle(M)
    A[A > 1.] = 1.

    R = A*(np.cos(P - 2*pi/3)/2+0.5)
    G = A*(np.cos(P)/2+0.5)
    B = A*(np.cos(P + 2*pi/3)/2+0.5)
    
    C = np.dstack((R, G, B))
    plt.imshow(C)
    plt.show() 

@complim.register
def _(x: torch.Tensor) -> torch.Tensor:
    mAx = torch.amax(torch.abs(x))
    M = x/mAx
    A = torch.abs(M)
    P = torch.angle(M)
    A[A > 1.] = 1.

    R = A*(torch.cos(P - 2*pi/3)/2+0.5)
    G = A*(torch.cos(P)/2+0.5)
    B = A*(torch.cos(P + 2*pi/3)/2+0.5)
    
    C = torch.dstack((R, G, B))
    plt.imshow(C)
    plt.show()    




@singledispatch
def plot_in_GS(x: Union[np.ndarray, torch.Tensor]):
    # visualize a 2D phase distribution in gray scale (8 bit)
    raise NotImplementedError(
        f"Cannot visualize `x` for type: {type(x)}")

@plot_in_GS.register
def _(x: np.ndarray) -> np.ndarray:
    x = np.angle(np.exp(1j*x))
    plt.imshow(x, cmap="gray")
    plt.show()    

@plot_in_GS.register
def _(x: torch.Tensor) -> torch.Tensor:
    x = torch.angle(torch.exp(1j*x))
    plt.imshow(x, cmap="gray")
    plt.show()    




@singledispatch
def complim_subplot2(x: Union[np.ndarray, torch.Tensor]):
    # visualize two complex fields side by side. brightness = amplitude, colour = phase
    raise NotImplementedError(
        f"Cannot visualize `x` for type: {type(x)}")

@complim_subplot2.register
def _(x: np.ndarray, y: np.ndarray, titles: list) -> np.ndarray:
    mAx = np.amax(np.abs(x))
    M = x/mAx
    A = np.abs(M)
    P = np.angle(M)
    A[A > 1.] = 1.

    R = A*(np.cos(P - 2*pi/3)/2+0.5)
    G = A*(np.cos(P)/2+0.5)
    B = A*(np.cos(P + 2*pi/3)/2+0.5)
    
    C1 = np.dstack((R, G, B))

    mAx = np.amax(np.abs(y))
    M = y/mAx
    A = np.abs(M)
    P = np.angle(M)
    A[A > 1.] = 1.

    R = A*(np.cos(P - 2*pi/3)/2+0.5)
    G = A*(np.cos(P)/2+0.5)
    B = A*(np.cos(P + 2*pi/3)/2+0.5)
    
    C2 = np.dstack((R, G, B))

    C = [C1, C2]
    fig, axs = plt.subplots(1, 2)
    i = 0
    for ax, interp in zip(axs, titles):
        ax.imshow(C[i])
        ax.set_title(interp, fontsize=10)
        i = i+1
    plt.show()

@complim_subplot2.register
def _(x: torch.Tensor, y: torch.Tensor, titles: list) -> torch.Tensor:
    mAx = torch.amax(torch.abs(x))
    M = x/mAx
    A = torch.abs(M)
    P = torch.angle(M)
    A[A > 1.] = 1.

    R = A*(torch.cos(P - 2*pi/3)/2+0.5)
    G = A*(torch.cos(P)/2+0.5)
    B = A*(torch.cos(P + 2*pi/3)/2+0.5)
    
    C1 = torch.dstack((R, G, B))

    mAx = torch.amax(torch.abs(y))
    M = y/mAx
    A = torch.abs(M)
    P = torch.angle(M)
    A[A > 1.] = 1.

    R = A*(torch.cos(P - 2*pi/3)/2+0.5)
    G = A*(torch.cos(P)/2+0.5)
    B = A*(torch.cos(P + 2*pi/3)/2+0.5)
    
    C2 = torch.dstack((R, G, B))

    C = [C1, C2]
    fig, axs = plt.subplots(1, 2)
    i = 0
    for ax, interp in zip(axs, titles):
        ax.imshow(C[i])
        ax.set_title(interp, fontsize=10)
        i = i+1
    plt.show()