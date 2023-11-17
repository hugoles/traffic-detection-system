import argparse
from typing import Dict, List, Set, Tuple

import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import datetime

import supervision as sv

COLORS = sv.ColorPalette.default()

ZONE_IN_POLYGONS = [
    np.array([[1100, 850], [1100, 500], [1300, 500], [1300, 850]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[250, 400], [1350, 400], [1350, 399], [250, 399]]),
]

class_id_counts = {
    0: 'counts_busInter',
    1: 'counts_busMun',
    2: 'counts_car',
    3: 'counts_truck',
    4: 'counts_van'
}

class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        self.counts_car = 0
        self.counts_busMun = 0
        self.counts_busInter = 0
        self.counts_truck = 0
        self.counts_van = 0
        '''
        self.counts_busMun = self.counts_busMun if hasattr(self, 'counts_busMun') else 0
        self.counts_busInter = self.counts_busInter if hasattr(self, 'counts_busInter') else 0
        self.counts_car = self.counts_car if hasattr(self, 'counts_car') else 0
        self.counts_truck = self.counts_truck if hasattr(self, 'counts_truck') else 0
        self.counts_van = self.counts_van if hasattr(self, 'counts_van') else 0
        '''

        self.unique_detect_ids: Dict[int, Set[int]] = {}


    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
                    for detect_id in detections_out_zone.class_id:
                        if detect_id not in self.unique_detect_ids.get(tracker_id, set()):
                            self.unique_detect_ids.setdefault(tracker_id, set()).add(detect_id)
                            count_attribute = class_id_counts.get(detect_id)
                            if count_attribute:
                                setattr(self, count_attribute, getattr(self, count_attribute) + 1)


        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)
        
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()


    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    count_car = self.detections_manager.counts_car
                    count_busM = self.detections_manager.counts_busMun
                    count_busI = self.detections_manager.counts_busInter
                    count_truck = self.detections_manager.counts_truck
                    count_van = self.detections_manager.counts_van
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text="Vehicles: " + str(count) + 
                        "| Cars: " + str(count_car) + 
                        "| BusesM: " + str(count_busM) + 
                        "| BusesI: " + str(count_busI) + 
                        "| Trucks: " + str(count_truck) + 
                        "| Vans: " + str(count_van),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()

    model = YOLO(args.source_weights_path)
    print(model.model.names)
    
    e = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    video = os.path.basename(args.target_video_path)
    model = os.path.basename(args.source_weights_path)
    
    with open('counter.csv', 'a') as f:
        for key, value in processor.detections_manager.counts.items():
            for key2, value2 in value.items():
                f.write(f'{e},{video},{model},{len(value2)},{processor.detections_manager.counts_car},{processor.detections_manager.counts_busMun},{processor.detections_manager.counts_busInter},{processor.detections_manager.counts_truck},{processor.detections_manager.counts_van}\n')
