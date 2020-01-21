"""microdraw.py: functions for working with MicroDraw"""

import urllib.request as urlreq
import hashlib
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.draw import polygon, polygon_perimeter
import nibabel as nib

def download_project_definition(project, token):
  '''Download a project. A project can contain several datasets'''

  url = "http://microdraw.pasteur.fr/project/json/%s?token=%s"%(project, token)
  res = urlreq.urlopen(url)
  txt = res.read()
  prj = json.loads(txt)

  return prj

def download_dataset_definition(source):
  '''Download a dataset. A dataset can contain several slices, each with
     several regions'''

  res = urlreq.urlopen(source)
  txt = res.read()
  prj = json.loads(txt)

  return {
      "pixelsPerMeter": prj["pixelsPerMeter"],
      "numSlices": len(prj["tileSources"]),
      "project": prj
      }

def download_all_regions_from_dataset_slice(source, project, slce, token):
  '''Download all regions in a dataset slice'''

  url = "http://microdraw.pasteur.fr/api?source=%s&project=%s&slice=%s&token=%s"%(
      source,
      project,
      slce,
      token)
  res = urlreq.urlopen(url)
  txt = res.read()
  conts = json.loads(txt)

  return conts

def download_all_regions_from_dataset(source, project, token):
  '''Download all regions in all slices in a dataset'''

  dataset = download_dataset_definition(source)
  dataset["slices"] = []
  for i in tqdm(range(dataset["numSlices"])):
    dataset["slices"].append(download_all_regions_from_dataset_slice(source, project, i, token))
  return dataset

def get_points_from_segment(seg):
  '''get points from segment'''

  points = []
  for i in range(seg.shape[0]):
    seg[i] = np.array(seg[i])
    if type(seg[i][0]).__name__ == "ndarray":
      # flatten curve segments
      points.append([seg[i][0][0], seg[i][0][1]])
    else:
      # line segment
      points.append([seg[i][0], seg[i][1]])
  return points

def get_regions_from_dataset_slice(dataset):
  '''get regions from dataset slice'''

  regions = []
  for region in dataset:
    name = region['annotation']['name']
    path_type = region['annotation']['path'][0]
    if path_type == "Path":
      seg = np.asfortranarray(region['annotation']['path'][1]['segments'])
      points = get_points_from_segment(seg)
      regions.append((name, np.array(points)))
    elif path_type == "CompoundPath":
      children = region['annotation']['path'][1]['children']
      for child in children:
        if 'segments' in child[1]:
          segments = [np.asfortranarray(child[1]['segments'])]
          for seg in segments:
            points = get_points_from_segment(seg)
            regions.append((name, np.array(points)))
  return regions

def draw_all_dataset(dataset, ncol=13, width=800, alpha=0.5, path=None):
  '''draw all dataset'''

  plt.figure(figsize=(25, 10))
  i = 0
  j = 0
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region in regions:
      color = color_from_string(name)
      plt.fill(region[:, 0]+i*width, -region[:, 1]-j*width, alpha=alpha, c=color)
      plt.text((i+0.5)*width, -(j+1)*width, str(slce), alpha=0.5)
    i += 1
    if i >= ncol:
      i = 0
      j += 1
  plt.axis('equal')
  if path:
    plt.savefig(path)

def dataset_as_volume(dataset):
  '''get dataset as a volume'''

  verts = []
  eds = []
  neds = 0
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for _, region in regions:
      verts.extend([(x, y, slce) for x, y in region])
      eds.extend([(neds+i, neds+(i+1)%len(region)) for i in range(len(region))])
      neds = len(eds)
  return verts, eds

def save_dataset_as_text_mesh(dataset, path):
  '''save dataset as a text mesh'''

  verts, eds = dataset_as_volume(dataset)
  mesh = "%i 0 %i"%(len(verts), len(eds))
  verts_str = "\n".join(["%f %f %f"%(x*0.1, y*0.1, z*1.25) for x, y, z in verts])
  eds_str = "\n".join(["%i %i"%(i, j) for i, j in eds])
  mesh = "\n".join((mesh, verts_str, eds_str))

  file = open(path, 'w')
  file.write(mesh)

def dataset_to_nifti(dataset, voxdim=[0.1, 0.1, 1.25], region_name=None):
  '''convert dataset to nifti volume'''

  verts, _ = dataset_as_volume(dataset)
  vmin, vmax = np.min(verts, axis=0), np.max(verts, axis=0)
  vmin = np.floor(vmin)
  vmax = np.ceil(vmax)
  size = vmax-vmin
  size[2] = dataset["numSlices"]
  img = np.zeros([int(x) for x in size], 'uint8')
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region in regions:
      if region_name is None or (region_name is not None and name == region_name):
        try:
          rows, cols = polygon(region[:, 0]-vmin[0], region[:, 1]-vmin[1], img.shape)
          img[rows, cols, slce] = 255
          rows, cols = polygon_perimeter(region[:, 0]-vmin[0], region[:, 1]-vmin[1], img.shape)
          img[rows, cols, slce] = 0
        except:
          continue
  affine = np.eye(4)
  affine[0, 0] = voxdim[0]
  affine[1, 1] = voxdim[1]
  affine[2, 2] = voxdim[2]
  nii = nib.Nifti1Image(img, affine=affine)

  return nii

def save_dataset_as_nifti(dataset, path, voxdim=[0.1, 0.1, 1.25], region_name=None):
  '''save dataset as a nifti volume'''

  nii = dataset_to_nifti(dataset, voxdim, region_name)
  nib.save(nii, path)

def save_dataset(data, path):
  '''save dataset in json format'''

  with open(path, 'w') as outfile:
    json.dump(data, outfile)

def color_from_string(my_string):
  '''create a random color based on a hash of the input string'''

  sha = hashlib.sha256()
  sha.update(my_string.encode())
  return "#" + sha.hexdigest()[:6]
