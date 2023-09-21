package com.example.realtimeimagerecognitionapp
//------------------------------ Nidy-2.1 ----------------------------------//

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color.rgb
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType.CV_8UC3
import org.opencv.core.Mat
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.roundToInt

class ImageAnalyze(context: Context) : ImageAnalysis.Analyzer {

    private lateinit var listener: OnAnalyzeListener    // 用于更新View的自定义侦听器
    private var lastAnalyzedTimestamp = 0L

    ///网络模型的模型加载

    //private val gpuFlag = true // gpu/cpu

    private val night2day = Module.load(getAssetFilePath(context, "mobile-htanh-01.pt"))

    // for YUV -> Bitmap
    private val converter = YuvToRgbConverter(context)

    interface OnAnalyzeListener {
        fun getDayResult(dayImage: Bitmap)
        fun getFPSResult(fpsShow: String)
        fun getLoc(fakeLoc: String)
        fun getWarning(warningText: String)
    }



    override fun analyze(image: ImageProxy, rotationDegrees: Int) {


            /// Get input for night2day. Overall process ->>  YUV -> Bitmap -> Tensor
            val inputTensor = getInputTensor(image, rotationDegrees)
            /*
            val inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(image.image, rotationDegrees,
                224,
                224,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB)*/

            val inputWidth = inputTensor.shape()[3].toInt()
            val inputHeight = inputTensor.shape()[2].toInt()

            // inference start time
            val currentTimestamp = System.currentTimeMillis()
            lastAnalyzedTimestamp = currentTimestamp

            /// 用已学习模型进行推理
            val outputTensor = night2day.forward(IValue.from(inputTensor)).toTensor()

            // Compute the FPS of the inference process
            val fpsShow = getFPS()
            val fakeLoc = getFakeLoc()
            val fakeWarning = getFakeWarning()

            // FloatArray -> Bitmap
            val dayImgBit =
                floatArrayToBitmap(
                outputTensor.dataAsFloatArray,
                inputWidth,
                inputHeight)
            // the night2day result need to be displayed)

            // 更新View。
            listener.getDayResult(dayImgBit)
            listener.getFPSResult(fpsShow)
            listener.getLoc(fakeLoc)
            listener.getWarning(fakeWarning)
    }

    //// 从asset文件获取路径的函数
    private fun getAssetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }

    fun setOnAnalyzeListener(listener: OnAnalyzeListener){
        this.listener = listener
    }

    private fun floatArrayToBitmap(floatArray: FloatArray, width: Int, height: Int) : Bitmap {

        // Create empty bitmap in RGBA format
        val bmp: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height * 4)

        // mapping smallest value to 0 and largest value to 255
        val maxValue = floatArray.max() ?: 1.0f
        val minValue = floatArray.min() ?: -1.0f
        val delta = maxValue-minValue

        // Define if float min..max will be mapped to 0..255 or 255..0
        val conversion = { v: Float -> ((v-minValue)/delta*255.0f).roundToInt()}

        // copy each value from float array to RGB channels and set alpha channel
        for (i in 0 until width * height) {
            val r = conversion(floatArray[i])
            val g = conversion(floatArray[i + width * height])
            val b = conversion(floatArray[i + 2 * width * height])
            pixels[i] = rgb(r, g, b)
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)

        return bmp
    }

    private fun floatArrayToBitmapX(floatArray: FloatArray, width: Int, height: Int) : Bitmap {

        // Create empty bitmap in RGBA format
        val bmp: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height * 4)

        // mapping smallest value to 0 and largest value to 255
        val maxValue = floatArray.max() ?: 1.0f
        val minValue = floatArray.min() ?: -1.0f
        val delta = maxValue-minValue

        // Define if float min..max will be mapped to 0..255 or 255..0
        val conversion = { v: Float -> ((v-minValue)/delta*255.0f).roundToInt()}

        // copy each value from float array to RGB channels and set alpha channel
        for (i in 0 until width * height) {
            val r = conversion(floatArray[i * 3])
            val g = conversion(floatArray[i * 3 + 1])
            val b = conversion(floatArray[i * 3 + 2])
            pixels[i] = rgb(r, g, b)
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)

        return bmp
    }

    private fun getFPS(): String {
        val now = System.currentTimeMillis()
        val delta = now - lastAnalyzedTimestamp
        val fps = 1000f / delta
        val fpsLog = "FPS: ${"%.02f".format(fps)} "
        val pipelineLog = "Inference: ${"%d".format(delta)}ms"
        Log.d(TAG, fpsLog)
        Log.d(TAG, pipelineLog)
        return fpsLog + pipelineLog
    }

    private fun getFakeLoc(): String {
        val randomN = (0..10).random()
        val randomE = (0..10).random()
        return "${"%.02f".format(120 + (randomN - 5) / 100.0)}E, " +
                "${"%.02f".format(30 + (randomE - 5) / 100.0)}N"
    }

    private fun getFakeWarning(): String {
        val random = (0..10).random()
        return if(random < 3){ "OBJECT APPROACHING!!!" }
        else{ "No danger" }
    }



    private fun getInputTensor(image: ImageProxy, rotationDegrees: Int): Tensor {


        /// Overall process ->>  yuv -> Bitmap -> Tensor
        val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        converter.yuvToRgb(image.image!!, bitmap)

        // create matrix for the manipulation
        val matrix = Matrix()

        // rotate the Bitmap
        matrix.postRotate(rotationDegrees.toFloat())

        val width: Int = bitmap.width
        val height: Int = bitmap.height
        val newWidth = 240
        val newHeight = 240

        // calculate the scale - in this case = 0.4f
        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height

        matrix.postScale(scaleWidth, scaleHeight)

        // recreate the new Bitmap
        var resizedBitmap = Bitmap.createBitmap(
            bitmap, 0, 0,
            width, height, matrix, true
        )

        // use CLAHE
        resizedBitmap = applyCLAHE(resizedBitmap)


        return TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            NORM_MEAN_RAB_CUSTOM,
            NORM_STD_RAB_CUSTOM // No need to normalize for night2day process
        )

    }

    private fun applyCLAHE(img: Bitmap): Bitmap {
        val result: Bitmap = Bitmap.createBitmap(img.width, img.height, Bitmap.Config.ARGB_8888)
        val mIn = Mat(img.height, img.width, CV_8UC3)
        Utils.bitmapToMat(img, mIn)
        val clahe: CLAHE = Imgproc.createCLAHE()
        clahe.clipLimit = 2.0
        val labIn: List<Mat> = ArrayList()
        val labOut: List<Mat> = ArrayList()

        Core.split(mIn, labIn)
        Core.split(mIn, labOut)

        for (i in 0..2){
            clahe.apply(labIn[i], labOut[i])
            clahe.collectGarbage()
        }
        val mOut = Mat(img.height, img.width, CV_8UC3)
        Core.merge(ArrayList(listOf(labOut[0], labOut[1], labOut[2])), mOut)
        Utils.matToBitmap(mOut, result)
        return result
    }

    companion object {
        val TAG = MainActivity::class.java.simpleName
        val NORM_MEAN_RAB_CUSTOM = floatArrayOf(0.0f, 0.0f, 0.0f)
        val NORM_STD_RAB_CUSTOM = floatArrayOf(1.0f, 1.0f, 1.0f)
    }

}






