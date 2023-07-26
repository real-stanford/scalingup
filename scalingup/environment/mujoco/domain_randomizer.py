import dataclasses
import os
from typing import Dict, List, Tuple
from dm_control.mjcf import RootElement
import numpy as np
from scalingup.algo.virtual_grid import Point3D
import torchvision
from PIL import Image
import colorsys


@dataclasses.dataclass
class DomainRandomizationConfig:
    headlight_diffuse: Tuple[float, float]
    headlight_ambient: Tuple[float, float]
    headlight_specular: Tuple[float, float]
    material_shininess: Tuple[float, float]

    num_directional_lights: Tuple[int, int]
    direction_light_diffuse: Tuple[float, float]
    direction_light_specular: Tuple[float, float]
    light_pos: Tuple[Point3D, Point3D]

    light_lookat: Tuple[Point3D, Point3D]

    add_wall_texture_prob: float
    add_floor_texture_prob: float
    dtd_root: str

    wall_hsv: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    floor_hsv: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    table_hsv: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

    camera_pos: Dict[str, Tuple[Point3D, Point3D]]

    camera_fovy: Dict[str, Tuple[float, float]]


dtd_cache: List[str] = []


def sample_random_texture(
    dtd_root: str,
    np_random: np.random.RandomState,
):
    global dtd_cache
    if len(dtd_cache) == 0:
        torchvision.datasets.DTD(root=dtd_root, download=True, split="train")
        dtd_cache = []
        for split in ["train", "val", "test"]:
            dtd_cache.extend(
                [
                    path.strip()
                    for path in open(
                        os.path.join(dtd_root, f"dtd/dtd/labels/{split}1.txt"), "r"
                    ).readlines()
                ]
            )

        dtd_cache = sorted(set(dtd_cache))
    # pick random texture

    jpg_path = os.path.join(dtd_root, "dtd/dtd/images/", np_random.choice(dtd_cache))
    png_path = jpg_path.split(".")[0] + ".png"
    Image.open(jpg_path).save(png_path)
    return png_path


def find_mat(root: RootElement, name: str):
    for mat in root.asset.find_all("material"):
        if mat.name == name:
            return mat
    return None


def find_cam(root: RootElement, name: str):
    for cam in root.worldbody.find_all("camera"):
        if cam.name == name:
            return cam
    return None


def add_texture_to_material(
    texture_name: str,
    texture_png_path: str,
    material_name: str,
    world_model: RootElement,
    np_random: np.random.RandomState,
    texrepeat: str = "1 1",
):
    world_model.asset.add(
        "texture",
        name=texture_name,
        type="2d",
        file=texture_png_path,
        vflip="true" if np_random.uniform() < 0.5 else "false",
        hflip="true" if np_random.uniform() < 0.5 else "false",
    )
    mat = find_mat(root=world_model, name=material_name)
    assert mat is not None
    mat.texture = texture_name
    mat.texrepeat = texrepeat
    mat.texuniform = "true"


def domain_randomize(
    world_model: RootElement,
    config: DomainRandomizationConfig,
    np_random: np.random.RandomState,
):
    visual = world_model.visual
    diffuse = np_random.uniform(*config.headlight_diffuse)
    specular = np_random.uniform(*config.headlight_specular)
    ambient = np_random.uniform(*config.headlight_ambient)

    visual.headlight.ambient = f"{ambient} {ambient} {ambient}"
    visual.headlight.diffuse = f"{diffuse} {diffuse} {diffuse}"
    visual.headlight.specular = f"{specular} {specular} {specular}"

    num_lights = np_random.randint(*config.num_directional_lights)
    for _ in range(num_lights):
        diffuse = np_random.uniform(*config.direction_light_diffuse)
        specular = np_random.uniform(*config.direction_light_specular)
        pos = np_random.uniform(
            low=np.array(config.light_pos[0]), high=np.array(config.light_pos[1])
        )
        lookat_pos = np_random.uniform(
            low=np.array(config.light_lookat[0]), high=np.array(config.light_lookat[1])
        )
        direction = lookat_pos - pos
        direction = direction / np.linalg.norm(direction)
        world_model.worldbody.add(
            "light",
            directional="true",
            mode="fixed",
            active="true",
            ambient="0 0 0 ",
            diffuse=f"{diffuse} {diffuse} {diffuse}",
            specular=f"{specular} {specular} {specular}",
            pos=f"{pos[0]} {pos[1]} {pos[2]}",
            dir=f"{direction[0]} {direction[1]} {direction[2]}",
        )
    if np_random.uniform() < config.add_wall_texture_prob:
        png_path = sample_random_texture(dtd_root=config.dtd_root, np_random=np_random)
        add_texture_to_material(
            texture_name="wall_texture",
            texture_png_path=png_path,
            material_name="wall",
            world_model=world_model,
            texrepeat="0.5 0.5",
            np_random=np_random,
        )
    if np_random.uniform() < config.add_floor_texture_prob:
        png_path = sample_random_texture(dtd_root=config.dtd_root, np_random=np_random)
        add_texture_to_material(
            texture_name="floor",
            texture_png_path=png_path,
            material_name="floor",
            world_model=world_model,
            texrepeat="0.5 0.5",
            np_random=np_random,
        )

    hue = np_random.uniform(*config.wall_hsv[0])
    saturation = np_random.uniform(*config.wall_hsv[1])
    value = np_random.uniform(*config.wall_hsv[2])
    wall_mat = find_mat(root=world_model, name="wall")
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    wall_mat.rgba = f"{red} {green} {blue} 1"

    hue = np_random.uniform(*config.floor_hsv[0])
    saturation = np_random.uniform(*config.floor_hsv[1])
    value = np_random.uniform(*config.floor_hsv[2])
    floor_mat = find_mat(root=world_model, name="floor")
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    floor_mat.rgba = f"{red} {green} {blue} 1"

    hue = np_random.uniform(*config.table_hsv[0])
    saturation = np_random.uniform(*config.table_hsv[1])
    value = np_random.uniform(*config.table_hsv[2])
    table_mat = find_mat(root=world_model, name="table")
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    table_mat.rgba = f"{red} {green} {blue} 1"

    for cam_name, cam_pos_bounds in config.camera_pos.items():
        pos = np_random.uniform(
            low=np.array(cam_pos_bounds[0]), high=np.array(cam_pos_bounds[1])
        )
        cam = find_cam(root=world_model, name=cam_name)
        assert cam is not None
        new_pos = cam.pos + pos
        cam.pos = f"{new_pos[0]} {new_pos[1]} {new_pos[2]}"
    for cam_name, cam_fovy in config.camera_fovy.items():
        fovy = np_random.uniform(*cam_fovy)
        cam = find_cam(root=world_model, name=cam_name)
        assert cam is not None
        cam.fovy = fovy
    return world_model


"""

reflectance: real, “0”

This attribute should be in the range [0 1]. If the value is greater than 0,
and the material is applied to a plane or a box geom, the renderer will
simulate reflectance. The larger the value, the stronger the reflectance.
For boxes, only the face in the direction of the local +Z axis is reflective.
Simulating reflectance properly requires ray-tracing which cannot (yet) be
done in real-time. We are using the stencil buffer and suitable projections
instead. Only the first reflective geom in the model is rendered as such.
This adds one extra rendering pass through all geoms, in addition to the
extra rendering pass added by each shadow-casting light.

"""
