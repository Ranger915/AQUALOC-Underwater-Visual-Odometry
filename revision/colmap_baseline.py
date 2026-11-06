from common import *
sys.path.append(str(ROOT/'revision_deps'))
import pycolmap as pc
work=CACHE/'colmap_sequential';work.mkdir(exist_ok=True);db=work/'database.db';model=work/'models';model.mkdir(exist_ok=True)
read=pc.ImageReaderOptions();read.camera_model='OPENCV';read.camera_params=','.join(str(x) for x in [K[0,0],K[1,1],K[0,2],K[1,2],*DIST])
extract=pc.FeatureExtractionOptions();extract.num_threads=2;extract.use_gpu=False;extract.sift.max_num_features=1500
matching=pc.FeatureMatchingOptions();matching.num_threads=2;matching.use_gpu=False
pairing=pc.SequentialPairingOptions();pairing.num_threads=2;pairing.overlap=10;pairing.quadratic_overlap=True;pairing.loop_detection=False
mapping=pc.IncrementalPipelineOptions();mapping.num_threads=2;mapping.random_seed=20261105;mapping.mapper.random_seed=20261105;mapping.triangulation.random_seed=20261105
mapping.ba_refine_focal_length=False;mapping.ba_refine_principal_point=False;mapping.ba_refine_extra_params=False;mapping.mapper.abs_pose_refine_focal_length=False;mapping.mapper.abs_pose_refine_extra_params=False
mapping.extract_colors=False
dump(OUT/'colmap_protocol.json',dict(version=pc.__version__,images=3385,preprocessing='raw',camera='OPENCV shared known calibration',reader=read.todict(),extraction=extract.todict(),matching=matching.todict(),pairing=pairing.todict(),mapping=mapping.todict(),comparison_scope='offline sequential SfM baseline, not online VO or independent physical ground truth'))
if not (work/'features.done').exists():
 print('EXTRACT',flush=True);pc.extract_features(str(db),str(IMAGES),camera_mode=pc.CameraMode.SINGLE,camera_model='OPENCV',reader_options=read,extraction_options=extract,device=pc.Device.cpu);(work/'features.done').write_text('complete')
if not (work/'matches.done').exists():
 print('MATCH',flush=True);pc.match_sequential(str(db),matching_options=matching,pairing_options=pairing,device=pc.Device.cpu);(work/'matches.done').write_text('complete')
print('MAP',flush=True);models=pc.incremental_mapping(str(db),str(IMAGES),str(model),options=mapping)
components=[];largest=None
for idx,rec in models.items():
 components.append(dict(component=idx,registered_images=rec.num_reg_images(),points=rec.num_points3D()))
 if largest is None or rec.num_reg_images()>largest.num_reg_images():largest=rec
dump(OUT/'colmap_components.json',components)
if largest is not None:
 rr=[]
 for img in largest.images.values():
  if not img.has_pose:continue
  frame=int(''.join(c for c in Path(img.name).stem if c.isdigit()));Tw=img.cam_from_world().inverse();T=Tw.matrix();r=dict(frame=frame,fresh=1,reset=0,cumulative_resets=0,cumulative_unknown_frames=0)
  r.update({f'p{k}':float(T[k,3]) for k in range(3)});r.update({f'R{i}{j}':float(T[i,j]) for i in range(3) for j in range(3)});rr.append(r)
 writecsv(OUT/'vo/COLMAP_sequential.csv',sorted(rr,key=lambda r:r['frame']))
 largest.write_text(str(work/'native_text')) if (work/'native_text').mkdir(exist_ok=True) is None else None
dump(OUT/'colmap_baseline.done.json',dict(components=components,status='complete',comparison='largest component, all components disclosed'))
print('COLMAP DONE',components,flush=True)
