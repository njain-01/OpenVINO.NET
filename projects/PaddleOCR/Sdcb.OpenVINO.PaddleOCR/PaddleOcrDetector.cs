using Clipper2Lib;
using OpenCvSharp;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using Sdcb.OpenVINO.PaddleOCR.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.CompilerServices;

namespace Sdcb.OpenVINO.PaddleOCR;

/// <summary>
/// The PaddleOcrDetector class is responsible for detecting text regions in an image using PaddleOCR.
/// It implements IDisposable to manage memory resources.
/// </summary>
public class PaddleOcrDetector : IDisposable
{
    static readonly Vec3f meanValues = new(
            255.0f * 0.485f,
            255.0f * 0.456f,
            255.0f * 0.406f);

    static readonly Vec3f stdValues = new(
        1 / (255.0f * 0.229f),
        1 / (255.0f * 0.224f),
        1 / (255.0f * 0.225f));

    readonly CompiledModel _compiledModel;

    /// <summary>
    /// Gets or sets the maximum size for resizing the input image.
    /// <para>Note: this property is invalid when <see cref="IsDynamicGraph"/> = <c>false</c></para>
    /// </summary>
    public int? MaxSize { get; set; } = 1536;

    /// <summary>Gets or sets the size for dilation during preprocessing.</summary>
    public int? DilatedSize { get; set; } = 2;

    /// <summary>Gets or sets the score threshold for filtering out possible text boxes.</summary>
    public float? BoxScoreThreahold { get; set; } = 0.6f;

    /// <summary>Gets or sets the threshold to binarize the text region.</summary>
    public float? BoxThreshold { get; set; } = 0.3f;

    /// <summary>Gets or sets the minimum size of the text boxes to be considered as valid.</summary>
    public int MinSize { get; set; } = 3;

    /// <summary>Gets or sets the ratio for enlarging text boxes during post-processing.</summary>
    public float UnclipRatio { get; set; } = 2.0f;

    /// <summary>
    /// Gets the static size of the input image for network infer.
    /// </summary>
    /// <remarks>
    /// If this property is null, network can work with image of any size and h/w ratio (dynamic).
    /// Otherwise, network works with fixed size image (static).
    /// </remarks>
    public Size? StaticShapeSize { get; } = null;

    /// <summary>
    /// Gets a value indicating whether the network uses dynamic graph.
    /// </summary>
    /// <value>
    ///   <c>true</c> if network uses dynamic graph; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// A graph can be static or dynamic. Static graphs have a fixed structure determined before execution, while dynamic graphs have an undefined structure that emerges during execution. 
    /// </remarks>
    public bool IsDynamicGraph => !StaticShapeSize.HasValue;

    /// <summary>
    /// Initializes a new instance of the PaddleOcrDetector class with the provided DetectionModel and PaddleConfig.
    /// </summary>
    /// <param name="model">The DetectionModel to use.</param>
    /// <param name="options">The device and configure of the PaddleConfig, pass null to using model's DefaultDevice.</param>
    /// <param name="staticShapeSize">
    /// The static size of the input image for network infer, 
    /// <para>If this property is null, network can work with image of any size and h/w ratio (dynamic).</para>
    /// <para>Otherwise, network works with fixed size image (static).</para>
    /// </param>
    public PaddleOcrDetector(DetectionModel model, 
        DeviceOptions? options = null, 
        Size? staticShapeSize = null)
    {
        if (staticShapeSize != null)
        {
            StaticShapeSize = new(
                32 * Math.Ceiling(1.0 * staticShapeSize.Value.Width / 32),
                32 * Math.Ceiling(1.0 * staticShapeSize.Value.Height / 32));
        }

        _compiledModel = model.CreateCompiledModel(options, afterReadModel: m =>
        {
            if (model.Version != ModelVersion.V4)
            {
                m.ReshapePrimaryInput(new PartialShape(1, 3, Dimension.Dynamic, Dimension.Dynamic));
            }
        }, prePostProcessing: (m, ppp) =>
        {
            using PreProcessInputInfo ppii = ppp.Inputs.Primary;
            ppii.TensorInfo.Layout = Layout.NHWC;
            ppii.ModelInfo.Layout = Layout.NCHW;
        }, afterBuildModel: m =>
        {
            if (StaticShapeSize != null)
            {
                m.ReshapePrimaryInput(new PartialShape(1, StaticShapeSize.Value.Height, StaticShapeSize.Value.Width, 3));
            }
            else if (model.Version != ModelVersion.V4)
            {
                m.ReshapePrimaryInput(new PartialShape(1, Dimension.Dynamic, Dimension.Dynamic, 3));
            }
        });
    }

    /// <summary>
    /// Disposes the PaddlePredictor instance.
    /// </summary>
    public void Dispose()
    {
        _compiledModel.Dispose();
    }

    /// <summary>
    /// Draws detected rectangles on the input image.
    /// </summary>
    /// <param name="src">Input image.</param>
    /// <param name="rects">Array of detected rectangles.</param>
    /// <param name="color">Color of the rectangles.</param>
    /// <param name="thickness">Thickness of the rectangle lines.</param>
    /// <returns>A new image with the detected rectangles drawn on it.</returns>
    public static Mat Visualize(Mat src, RotatedRect[] rects, Scalar color, int thickness)
    {
        Mat clone = src.Clone();
        clone.DrawContours(rects.Select(x => x.Points().Select(x => (Point)x)), -1, color, thickness);
        return clone;
    }

    public List<Point2f[]> RunNew(Mat src)
    {
        using Mat pred = RunRaw(src, out Size resizedSize);
        //pred.SaveImage("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-det-pred.png");
        using Mat cbuf = new();
        {
            using Mat roi = pred[0, resizedSize.Height, 0, resizedSize.Width];
            roi.ConvertTo(cbuf, MatType.CV_8UC1, 255);
        }
        using Mat dilated = new();
        {
            using Mat binary = BoxThreshold != null ?
                cbuf.Threshold((int)(BoxThreshold * 255), 255, ThresholdTypes.Binary) :
                cbuf;

            if (DilatedSize != null)
            {
                using Mat ones = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(DilatedSize.Value, DilatedSize.Value));
                Cv2.Dilate(binary, dilated, ones);
            }
            else
            {
                Cv2.CopyTo(binary, dilated);
            }
        }

        Point[][] contours = dilated.FindContoursAsArray(RetrievalModes.List, ContourApproximationModes.ApproxSimple);
        //// Serialize the sorted array to JSON
        //var json = JsonConvert.SerializeObject(contours, Newtonsoft.Json.Formatting.Indented);

        //// Write the JSON string to a file
        //File.WriteAllText("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-contours.json", json);
        Size size = src.Size();
        double scaleRate = 1.0 * src.Width / resizedSize.Width;

        //RotatedRect[] rects = contours
        //    .Where(x => BoxScoreThreahold == null || GetScore(x, pred) > BoxScoreThreahold)
        //    .Select(x => Cv2.MinAreaRect(x))
        //    .Where(x => x.Size.Width > MinSize && x.Size.Height > MinSize)
        //    .Select(rect =>
        //    {
        //        float minEdge = Math.Min(rect.Size.Width, rect.Size.Height);
        //        Size2f newSize = new(
        //            (rect.Size.Width + UnclipRatio * minEdge) * scaleRate,
        //            (rect.Size.Height + UnclipRatio * minEdge) * scaleRate);
        //        RotatedRect largerRect = new(rect.Center * scaleRate, newSize, rect.Angle);
        //        return largerRect;
        //    })
        //    .OrderBy(v => v.Center.Y)
        //    .ThenBy(v => v.Center.X)
        //    .ToArray();

        List<RotatedRect> filteredRects = new List<RotatedRect>();
        List<Point2f[]> point2Fs = new();
        List<float> scores = new();
        List<RotatedRect> conts = new();
        List<int> rejectedConts = new();
        int ind = 0;
        List<string> scoresList = new();
        foreach (var contour in contours)
        {
            try
            {
                var score = GetScore(contour, pred);
                if (BoxScoreThreahold == null || score > BoxScoreThreahold)
                {

                    // Get the minimum area rectangle
                    //RotatedRect rect = Cv2.MinAreaRect(contour);
                    var (bpoints, sside) = GetMiniBoxes(contour);

                    // Check if the rect size meets the minimum size criteria
                    if (!(sside < MinSize))
                    {
                        //Console.WriteLine(ind);
                        var points = Unclip(bpoints, UnclipRatio);
                        var (expbpoints, ssideExp) = GetMiniBoxes(points);
                        if (ssideExp < MinSize + 2)
                            continue;
                        TransformPoints(expbpoints, resizedSize.Width, resizedSize.Height, src.Width, src.Height);
                        try
                        {
                            //var largerRect = RotatedRect.FromThreeVertexPoints(expbpoints[0], expbpoints[1], expbpoints[2]);
                            //filteredRects.Add(largerRect);

                            point2Fs.Add(expbpoints);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e.Message);
                            //GetRotateCropImage(src, expbpoints).SaveImage($"C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-ov-det-imgs\\{ind}.png");
                        }
                    }
                    else
                    {
                        rejectedConts.Add(ind);
                    }
                }
                else
                {
                    rejectedConts.Add(ind);
                }
                ind++;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error");
            }
        }
        // First filtering based on BoxScoreThreahold
        //foreach (var contour in contours)
        //{
        //    var score = GetScore(contour, pred);
        //    RotatedRect rect2 = Cv2.MinAreaRect(contour);
        //    //scoresList.Add($"{ind}-->{score.ToString().Substring(0,5)}");
        //    conts.Add(rect2);
        //    if (BoxScoreThreahold == null || score > BoxScoreThreahold)
        //    {

        //        // Get the minimum area rectangle
        //        RotatedRect rect = Cv2.MinAreaRect(contour);
        //        var (bpoints, sside) = GetMiniBoxes(contour);
        //        //Cv2.BoxPoints()
        //        //conts.Add(rect);

        //        // Check if the rect size meets the minimum size criteria
        //        if (!(sside<MinSize))
        //        {
        //            scores.Add(score);
        //            // Calculate the new size
        //            float minEdge = Math.Min(rect.Size.Width, rect.Size.Height);
        //            Size2f newSize = new Size2f(
        //                (rect.Size.Width + UnclipRatio * minEdge) * scaleRate,
        //                (rect.Size.Height + UnclipRatio * minEdge) * scaleRate
        //            );

        //            // Create a larger rect with the new size
        //            RotatedRect largerRect = new RotatedRect(rect.Center * scaleRate, newSize, rect.Angle);
        //            var rectt = largerRect.BoundingRect();

        //            // Add the larger rect to the filtered list
        //            filteredRects.Add(largerRect);
        //        }
        //        else
        //        {
        //            rejectedConts.Add(ind);
        //        }
        //    }
        //    else
        //    {
        //        rejectedConts.Add(ind);
        //    }
        //    ind++;
        //}
        //Console.WriteLine($"Rejected conts {rejectedConts.Count}: {string.Join(" ", rejectedConts)}");
        //Console.WriteLine($"{string.Join("\n", scoresList)}");
        //var sortedRotatedRects = conts.OrderBy(r => r.Center.X).ToArray();

        //// Serialize the sorted array to JSON
        //var json = JsonConvert.SerializeObject(sortedRotatedRects, Newtonsoft.Json.Formatting.Indented);

        //// Write the JSON string to a file
        //File.WriteAllText("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-rects.json", json);


        //{
        //	using Mat demo = dilated.CvtColor(ColorConversionCodes.GRAY2RGB);
        //	demo.DrawContours(contours, -1, Scalar.Red);
        //	Image(demo).Dump();
        //}
        return point2Fs;
    }



    /// <summary>
    /// Runs the text box detection process on the input image.
    /// </summary>
    /// <param name="src">Input image.</param>
    /// <returns>An array of detected rotated rectangles representing text boxes.</returns>
    /// <exception cref="ArgumentException">Thrown when input image is empty.</exception>
    /// <exception cref="NotSupportedException">Thrown when input image channels are not 3 or 1.</exception>
    /// <exception cref="Exception">Thrown when PaddlePredictor run fails.</exception>
    public RotatedRect[] Run(Mat src)
    {
        using Mat pred = RunRaw(src, out Size resizedSize);
        //pred.SaveImage("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-det-pred.png");
        using Mat cbuf = new();
        {
            using Mat roi = pred[0, resizedSize.Height, 0, resizedSize.Width];
            roi.ConvertTo(cbuf, MatType.CV_8UC1, 255);
        }
        using Mat dilated = new();
        {
            using Mat binary = BoxThreshold != null ?
                cbuf.Threshold((int)(BoxThreshold * 255), 255, ThresholdTypes.Binary) :
                cbuf;

            if (DilatedSize != null)
            {
                using Mat ones = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(DilatedSize.Value, DilatedSize.Value));
                Cv2.Dilate(binary, dilated, ones);
            }
            else
            {
                Cv2.CopyTo(binary, dilated);
            }
        }

        Point[][] contours = dilated.FindContoursAsArray(RetrievalModes.List, ContourApproximationModes.ApproxSimple);
        //// Serialize the sorted array to JSON
        //var json = JsonConvert.SerializeObject(contours, Newtonsoft.Json.Formatting.Indented);

        //// Write the JSON string to a file
        //File.WriteAllText("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-contours.json", json);
        Size size = src.Size();
        double scaleRate = 1.0 * src.Width / resizedSize.Width;

        //RotatedRect[] rects = contours
        //    .Where(x => BoxScoreThreahold == null || GetScore(x, pred) > BoxScoreThreahold)
        //    .Select(x => Cv2.MinAreaRect(x))
        //    .Where(x => x.Size.Width > MinSize && x.Size.Height > MinSize)
        //    .Select(rect =>
        //    {
        //        float minEdge = Math.Min(rect.Size.Width, rect.Size.Height);
        //        Size2f newSize = new(
        //            (rect.Size.Width + UnclipRatio * minEdge) * scaleRate,
        //            (rect.Size.Height + UnclipRatio * minEdge) * scaleRate);
        //        RotatedRect largerRect = new(rect.Center * scaleRate, newSize, rect.Angle);
        //        return largerRect;
        //    })
        //    .OrderBy(v => v.Center.Y)
        //    .ThenBy(v => v.Center.X)
        //    .ToArray();

        List<RotatedRect> filteredRects = new List<RotatedRect>();
        List<float> scores = new();
        List<RotatedRect> conts = new();
        List<int> rejectedConts = new();
        int ind = 0;
        List<string> scoresList = new();
        foreach (var contour in contours)
        {
            try
            {
                var score = GetScore(contour, pred);
                if (BoxScoreThreahold == null || score > BoxScoreThreahold)
                {

                    // Get the minimum area rectangle
                    //RotatedRect rect = Cv2.MinAreaRect(contour);
                    var (bpoints, sside) = GetMiniBoxes(contour);

                    // Check if the rect size meets the minimum size criteria
                    if (!(sside < MinSize))
                    {
                        Console.WriteLine(ind);
                        var points = Unclip(bpoints, UnclipRatio);
                        var (expbpoints, ssideExp) = GetMiniBoxes(points);
                        if (ssideExp < MinSize + 2)
                            continue;
                        TransformPoints(expbpoints, resizedSize.Width, resizedSize.Height, src.Width, src.Height);
                        try
                        {
                            var largerRect = RotatedRect.FromThreeVertexPoints(expbpoints[0], expbpoints[1], expbpoints[2]);
                            filteredRects.Add(largerRect);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e.Message);
                            Console.WriteLine(e.HelpLink);
                            Console.WriteLine(e.StackTrace);
                            GetRotateCropImage(src, expbpoints).SaveImage($"C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-ov-det-imgs\\{ind}.png");
                        }
                    }
                    else
                    {
                        rejectedConts.Add(ind);
                    }
                }
                else
                {
                    rejectedConts.Add(ind);
                }
                ind++;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error");
            }
        }
        // First filtering based on BoxScoreThreahold
        //foreach (var contour in contours)
        //{
        //    var score = GetScore(contour, pred);
        //    RotatedRect rect2 = Cv2.MinAreaRect(contour);
        //    //scoresList.Add($"{ind}-->{score.ToString().Substring(0,5)}");
        //    conts.Add(rect2);
        //    if (BoxScoreThreahold == null || score > BoxScoreThreahold)
        //    {

        //        // Get the minimum area rectangle
        //        RotatedRect rect = Cv2.MinAreaRect(contour);
        //        var (bpoints, sside) = GetMiniBoxes(contour);
        //        //Cv2.BoxPoints()
        //        //conts.Add(rect);

        //        // Check if the rect size meets the minimum size criteria
        //        if (!(sside<MinSize))
        //        {
        //            scores.Add(score);
        //            // Calculate the new size
        //            float minEdge = Math.Min(rect.Size.Width, rect.Size.Height);
        //            Size2f newSize = new Size2f(
        //                (rect.Size.Width + UnclipRatio * minEdge) * scaleRate,
        //                (rect.Size.Height + UnclipRatio * minEdge) * scaleRate
        //            );

        //            // Create a larger rect with the new size
        //            RotatedRect largerRect = new RotatedRect(rect.Center * scaleRate, newSize, rect.Angle);
        //            var rectt = largerRect.BoundingRect();

        //            // Add the larger rect to the filtered list
        //            filteredRects.Add(largerRect);
        //        }
        //        else
        //        {
        //            rejectedConts.Add(ind);
        //        }
        //    }
        //    else
        //    {
        //        rejectedConts.Add(ind);
        //    }
        //    ind++;
        //}
        //Console.WriteLine($"Rejected conts {rejectedConts.Count}: {string.Join(" ", rejectedConts)}");
        //Console.WriteLine($"{string.Join("\n", scoresList)}");
        //var sortedRotatedRects = conts.OrderBy(r => r.Center.X).ToArray();

        //// Serialize the sorted array to JSON
        //var json = JsonConvert.SerializeObject(sortedRotatedRects, Newtonsoft.Json.Formatting.Indented);

        //// Write the JSON string to a file
        //File.WriteAllText("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-rects.json", json);

        // Sorting the filtered rectangles
        filteredRects.Sort((a, b) =>
        {
            int compareY = a.Center.Y.CompareTo(b.Center.Y);
            if (compareY != 0)
            {
                return compareY;
            }
            return a.Center.X.CompareTo(b.Center.X);
        });

        // Convert the list to an array
        RotatedRect[] rects = filteredRects.ToArray();

        //{
        //	using Mat demo = dilated.CvtColor(ColorConversionCodes.GRAY2RGB);
        //	demo.DrawContours(contours, -1, Scalar.Red);
        //	Image(demo).Dump();
        //}
        return rects;
    }

    /// <summary>
    /// Runs detection on the provided input image and returns the detected image as a <see cref="MatType.CV_32FC1"/> <see cref="Mat"/> object.
    /// </summary>
    /// <param name="src">The input image to run detection model on.</param>
    /// <param name="resizedSize">The returned image actuall size without padding.</param>
    /// <returns>the detected image as a <see cref="MatType.CV_32FC1"/> <see cref="Mat"/> object.</returns>
    public unsafe Mat RunRaw(Mat src, out Size resizedSize)
    {
        if (src.Empty())
        {
            throw new ArgumentException("src size should not be 0, wrong input picture provided?");
        }

        if (!(src.Channels() switch { 3 or 1 => true, _ => false }))
        {
            throw new NotSupportedException($"{nameof(src)} channel must be 3 or 1, provided {src.Channels()}.");
        }

        Mat padded = null!;
        if (IsDynamicGraph)
        {
            using Mat resized = MatResize(src, MaxSize);
            resizedSize = new Size(resized.Width, resized.Height);
            padded = MatPadding32(resized);
            //padded = resized.FastClone();
            //padded.SaveImage("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\resized_csharp_det.png");
        }
        else
        {
            using Mat resized = MatResize(src, StaticShapeSize!.Value);
            resizedSize = new Size(resized.Width, resized.Height);
            padded = resized.CopyMakeBorder(0, StaticShapeSize.Value.Height - resizedSize.Height, 0, StaticShapeSize.Value.Width - resizedSize.Width, BorderTypes.Constant, Scalar.Black);
        }

        Mat normalized;
        using (Mat _ = padded)
        {
            normalized = Normalize(padded);
            //normalized.SaveImage("C:\\Users\\NamanJain\\CV 4.20.0\\accuracy-gap\\csharp-det-norm.png");
        }

        using InferRequest ir = _compiledModel.CreateInferRequest();
        using (Mat _ = normalized)
        using (Tensor input = normalized.AsTensor())
        {
            ir.Inputs.Primary = input;
            ir.Run();
        }

        using (Tensor output = ir.Outputs[0])
        {
            Span<float> data = output.GetData<float>();
            NCHW shape = output.Shape.ToNCHW();

            return new Mat(shape.Height, shape.Width, MatType.CV_32FC1, data.ToArray());
        }
    }

    Mat GetRotateCropImage(Mat img, Point2f[] points)
    {
        // Compute width and height of the rotated rectangle using Euclidean distance
        int imgCropWidth = (int)Math.Max(
            Distance(points[0], points[1]),
            Distance(points[2], points[3])
        );
        int imgCropHeight = (int)Math.Max(
            Distance(points[0], points[3]),
            Distance(points[1], points[2])
        );

        // Destination points for perspective transform
        Point2f[] ptsStd = new Point2f[]
        {
            new Point2f(0, 0),
            new Point2f(imgCropWidth, 0),
            new Point2f(imgCropWidth, imgCropHeight),
            new Point2f(0, imgCropHeight)
        };

        // Get the perspective transform matrix
        Mat M = Cv2.GetPerspectiveTransform(points, ptsStd);

        // Apply perspective warp
        Mat dstImg = new Mat();
        Cv2.WarpPerspective(img, dstImg, M, new Size(imgCropWidth, imgCropHeight), InterpolationFlags.Cubic, BorderTypes.Replicate);

        // Rotate the image if the height/width ratio is greater than or equal to 1.5
        if (dstImg.Height * 1.0 / dstImg.Width >= 1.5)
        {
            Cv2.Transpose(dstImg, dstImg);
            Cv2.Flip(dstImg, dstImg, FlipMode.X);
        }

        return dstImg;
    }

    public static void TransformPoints(Point2f[] points, float width, float height, float destWidth, float destHeight)
    {
        for (int i = 0; i < points.Length; i++)
        {
            // Scale the coordinates
            float newX = points[i].X / width * destWidth;
            float newY = points[i].Y / height * destHeight;

            // Clip the coordinates to the destination dimensions
            newX = Math.Clamp((float)Math.Round(newX), 0, destWidth);
            newY = Math.Clamp((float)Math.Round(newY), 0, destHeight);

            // Update the point with new coordinates
            points[i] = new Point2f(newX, newY);
        }
    }

    public (Point2f[], float) GetMiniBoxes(Point[] contour)
    {
        // Convert Point[] to Point2f[]
        Point2f[] contour2f = Array.ConvertAll(contour, p => new Point2f(p.X, p.Y));

        // Compute the minimum area bounding box
        RotatedRect boundingBox = Cv2.MinAreaRect(contour2f);

        // Get the four corners of the bounding box
        Point2f[] points = Cv2.BoxPoints(boundingBox).OrderBy(p => p.X).ToArray();

        // Determine the index of the corners
        int index1 = 0, index2 = 1, index3 = 2, index4 = 3;

        if (points[1].Y > points[0].Y)
        {
            index1 = 0;
            index4 = 1;
        }
        else
        {
            index1 = 1;
            index4 = 0;
        }

        if (points[3].Y > points[2].Y)
        {
            index2 = 2;
            index3 = 3;
        }
        else
        {
            index2 = 3;
            index3 = 2;
        }

        // Arrange the points in the required order
        Point2f[] box = new Point2f[] { points[index1], points[index2], points[index3], points[index4] };

        // Return the points and the minimum side length of the bounding box
        return (box, Math.Min(boundingBox.Size.Width, boundingBox.Size.Height));
    }

    public static Point2f[] ConvertToRectangle(Point2f[] points)
    {
        if (points.Length != 4)
        {
            throw new ArgumentException("Exactly four points are required.");
        }

        // Extract the points
        Point2f p1 = points[0];
        Point2f p2 = points[1];
        Point2f p3 = points[2];
        Point2f p4 = points[3];

        // Find midpoints of opposite sides
        Point2f midpoint1 = new Point2f((p1.X + p4.X) / 2, (p1.Y + p4.Y) / 2);
        Point2f midpoint2 = new Point2f((p2.X + p3.X) / 2, (p2.Y + p3.Y) / 2);

        // Calculate the angle of the side
        double angle = Math.Atan2(p2.Y - p1.Y, p2.X - p1.X);

        // Calculate the lengths of sides
        double length1 = Distance(p1, p2);
        double length2 = Distance(p2, p3);

        // Create new rectangle points
        Point2f newP1 = new Point2f(midpoint1.X - (float)(length1 / 2 * Math.Cos(angle)), midpoint1.Y - (float)(length1 / 2 * Math.Sin(angle)));
        Point2f newP2 = new Point2f(midpoint1.X + (float)(length1 / 2 * Math.Cos(angle)), midpoint1.Y + (float)(length1 / 2 * Math.Sin(angle)));
        Point2f newP3 = new Point2f(midpoint2.X + (float)(length2 / 2 * Math.Cos(angle + Math.PI / 2)), midpoint2.Y + (float)(length2 / 2 * Math.Sin(angle + Math.PI / 2)));
        Point2f newP4 = new Point2f(midpoint2.X - (float)(length2 / 2 * Math.Cos(angle + Math.PI / 2)), midpoint2.Y - (float)(length2 / 2 * Math.Sin(angle + Math.PI / 2)));

        return new Point2f[] { newP1, newP2, newP3, newP4 };
    }

    private static double Distance(Point2f p1, Point2f p2)
    {
        return Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
    }

    public (Point2f[], float) GetMiniBoxes(Point2f[] contour2f)
    {
        // Compute the minimum area bounding box
        RotatedRect boundingBox = Cv2.MinAreaRect(contour2f);

        // Get the four corners of the bounding box
        Point2f[] points = Cv2.BoxPoints(boundingBox).OrderBy(p => p.X).ToArray();

        // Determine the index of the corners
        int index1 = 0, index2 = 1, index3 = 2, index4 = 3;

        if (points[1].Y > points[0].Y)
        {
            index1 = 0;
            index4 = 1;
        }
        else
        {
            index1 = 1;
            index4 = 0;
        }

        if (points[3].Y > points[2].Y)
        {
            index2 = 2;
            index3 = 3;
        }
        else
        {
            index2 = 3;
            index3 = 2;
        }

        // Arrange the points in the required order
        Point2f[] box = new Point2f[] { points[index1], points[index2], points[index3], points[index4] };

        // Return the points and the minimum side length of the bounding box
        return (box, Math.Min(boundingBox.Size.Width, boundingBox.Size.Height));
    }

    private static float GetScore(Point[] contour, Mat pred)
    {
        int width = pred.Width;
        int height = pred.Height;
        int[] boxX = contour.Select(v => v.X).ToArray();
        int[] boxY = contour.Select(v => v.Y).ToArray();

        int xmin = MathUtil.Clamp(boxX.Min(), 0, width - 1);
        int xmax = MathUtil.Clamp(boxX.Max(), 0, width - 1);
        int ymin = MathUtil.Clamp(boxY.Min(), 0, height - 1);
        int ymax = MathUtil.Clamp(boxY.Max(), 0, height - 1);

        Point[] rootPoints = contour
            .Select(v => new Point(v.X - xmin, v.Y - ymin))
            .ToArray();
        using Mat mask = new(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1, Scalar.Black);
        mask.FillPoly(new[] { rootPoints }, new Scalar(1));

        using Mat croppedMat = pred[ymin, ymax + 1, xmin, xmax + 1];
        float score = (float)croppedMat.Mean(mask).Val0;

        return score;
    }

    private static Mat MatResize(Mat src, int? maxSize)
    {
        if (maxSize == null) return src.FastClone();

        Size size = src.Size();
        int longEdge = Math.Max(size.Width, size.Height);
        double scaleRate = 1.0 * maxSize.Value / longEdge;

        //return scaleRate < 1.0 ?
        //    src.Resize(default, scaleRate, scaleRate) :
        //    src.FastClone();

        var resize_h =  scaleRate * size.Height;
        var resize_w = scaleRate * size.Width;
        double h32 = resize_h / 32;
        double w32 = resize_w / 32;
        resize_h = (int)(Math.Round(h32) * 32);
        resize_w = (int)(Math.Round(w32) * 32);

        if (scaleRate < 1.0)
        {
            Mat mat = new();
            Cv2.Resize(src, mat, new Size(resize_w, resize_h));
            return mat;
        }
        else
            return src.FastClone(); // TODO: handle padding in case of >1 scale rate
    }

    public static Path64 MakePoly(Point2f[] box)
    {
        Path64 p = new Path64(box.Length);
        for (int i = 0; i < box.Length; i++)
            p.Add(new Point64(box[i].X, box[i].Y));
        return p;
    }

    public static Point2f[] Unclip(Point2f[] box, float unclipRatio)
    {
        // Create a ClipperPath from the box points
        var polyPath = MakePoly(box);
        ClipperOffset offset = new ClipperOffset();

        // Calculate the unclipping distance
        double area = Clipper.Area(polyPath);
        var rect = Clipper.GetBounds(polyPath);
        double length = 2 * (rect.Width + rect.Height);
        double distance = area * unclipRatio / length;

        // Perform the offset
        offset.AddPath(polyPath, JoinType.Round, EndType.Polygon);
        var sln = new Paths64();
        offset.Execute(distance, sln);

        var expandedPoints = new List<Point2f>();
        foreach (var p in sln)
        {
            foreach(var p2 in p)
            {
                expandedPoints.Add(new Point2f(p2.X, p2.Y));
            }
        }

        return expandedPoints.ToArray();
    }

    private static Mat MatResize(Mat src, Size maxSize)
    {
        Size srcSize = src.Size();
        if (srcSize == maxSize)
        {
            return src.FastClone();
        }

        double scale = Math.Min(maxSize.Width / (double)srcSize.Width, maxSize.Height / (double)srcSize.Height);

        // Ensure the scale is never more than 1 (i.e., the image is never magnified)
        scale = Math.Min(scale, 1.0);

        if (scale == 1)
        {
            return src.FastClone();
        }
        else
        {
            // New size
            Size newSize = new((int)(scale * srcSize.Width), (int)(scale * srcSize.Height));

            // Set the resized image
            Mat resizedMat = new();

            Cv2.Resize(src, resizedMat, newSize);

            return resizedMat;
        }
    }

    private static Mat MatPadding32(Mat src)
    {
        Size size = src.Size();
        Size newSize = new(
            32 * Math.Ceiling(1.0 * size.Width / 32),
            32 * Math.Ceiling(1.0 * size.Height / 32));
        return src.CopyMakeBorder(0, newSize.Height - size.Height, 0, newSize.Width - size.Width, BorderTypes.Constant, Scalar.Black);
    }

    private static unsafe Mat Normalize(Mat src)
    {
        if (src.Type() != MatType.CV_32SC3)
        {
            src.ConvertTo(src, MatType.CV_32FC3);
        }

        int height = src.Height;
        int width = src.Width;
        int channel = src.Channels();
        float[] dstFloat = new float[width * height * channel];

        src.GetArray(out Vec3f[]? pixels);
        ref float srcFloat = ref Unsafe.As<Vec3f, float>(ref pixels[0]);

        fixed (float* pSrc = &srcFloat)
        fixed (float* pDst = &dstFloat[0])
        {
            Unsafe.CopyBlockUnaligned(pDst, pSrc, (uint)dstFloat.Length * sizeof(float));

            float* pointer = pDst;

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    *pointer = (*pointer - meanValues.Item0) * stdValues.Item0;
                    pointer += 1;
                    *pointer = (*pointer - meanValues.Item1) * stdValues.Item1;
                    pointer += 1;
                    *pointer = (*pointer - meanValues.Item2) * stdValues.Item2;
                    pointer += 1;
                }
            }
        }

        return new Mat(height, width, MatType.CV_32FC(channel), dstFloat);
    }
}
