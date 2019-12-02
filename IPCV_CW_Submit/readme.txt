> make dartsH
> ./dartsHough {image}

This is our final dartboard detector that makes use of strict
comparisons with expected dartboard features, and approximation
based on an incomplete feature set.
The output image is detected.jpg
(Best Performing - Stage 4 in report)

> make dartsHO
> ./dartsHoughO {image}

Our old dartboard detector that only made use of strict comparisons
with expected dartboard features.
(Stage 3 in report)

> make darts
> ./darts {image}

Dartboard detector that only uses Viola-Jones classifier.
The output image is dart_detected.jpg
(Stage 2 in report)

> make face
> ./face {image}

Face detector provided, with ground truths and detector result stats.
The output result is in face_detected.jpg

> make hough
> ./hough {image}

Generate 2d image of line hough space and circle hough space and thresholded image.
The output images are:
hough_line_thresh.jpg
hough_line.jpg
hough_circle_thresh.jpg
hough_circle.jpg
