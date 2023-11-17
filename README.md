# transit-detection
disc: This is a application forked by roboflow/supervision (roboflow/supervision/traffic_analysis/examples)
<div>
  <p>
    Transit detection is a project that use IA to detect, track and count vehicles
  </p>


https://github.com/hugoles/transit-detection/assets/67278688/57b8f158-3ad2-4628-be9c-2e2e70a183ce

This modified script can count, track, annotate, classify, and distinguish different classes of vehicles.

</div>


## ⚙️ run

```bash
python script.py \
--source_weights_path data/model4.pt \
--source_video_path data/output.mov \
--confidence_threshold 0.3 \
--iou_threshold 0.5 \
--target_video_path data/output_result.mov
```
