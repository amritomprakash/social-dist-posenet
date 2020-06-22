import tensorflow as tf
import cv2
import time
import argparse
import math
import posenet
import numpy
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def dist(x1,y1,x2,y2):
    return math.sqrt(((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            s1=[]
            d=0
            #cv2.putText(overlay_image,str(frame_count / (time.time() - start)),(320,320),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0))
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                #cv2.putText(overlay_image,str(pi),(int(keypoint_coords[pi,0,0]),int(keypoint_coords[pi,0,1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),)
                # print(keypoint_coords[pi, 5, :])
                # print(keypoint_coords[pi, 9, :])
                print(keypoint_coords[pi,0,0],keypoint_coords[pi,0,1])
                # 3 ft distance
                x1 = keypoint_coords[pi, 5, 0]
                y1 = keypoint_coords[pi, 5, 1]
                x2 = keypoint_coords[pi, 9, 0]
                y2 = keypoint_coords[pi, 9, 1]
                d = dist(x1,y1,x2,y2)
                s1.append(d)
            if not len(s1)==0:
              d=max(s1)
            #if not d==0:
             #print(d)
            arr = numpy.array(pose_scores)
            arr=arr[arr!=0]
            arr = arr.tolist()

            all_combinations = []

            combinations_object = itertools.combinations(range(len(arr)), 2)
            all_combinations = list(combinations_object)

            # if len(all_combinations) > 0:
            #  print(all_combinations)
            # # if len(all_combinations)>0:
            #    print(all_combinations[0][0] + all_combinations[0][1])

            if len(all_combinations) > 0:
              for i1 in range(len(all_combinations)):
                  c1=all_combinations[i1][0]
                  c2=all_combinations[i1][1]
                  if dist(keypoint_coords[c1,0,0],keypoint_coords[c1,0,1],keypoint_coords[c2,0,0],keypoint_coords[c2,0,1])>d:
                      cv2.putText(overlay_image,"violated",(320,320),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0))


            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()