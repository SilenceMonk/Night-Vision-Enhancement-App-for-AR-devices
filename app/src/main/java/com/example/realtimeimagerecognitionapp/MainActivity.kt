package com.example.realtimeimagerecognitionapp
// 注意看第88行 ->>
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.location.Location
import android.location.LocationManager
import android.os.Bundle
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors

private const val REQUEST_CODE_PERMISSIONS = 10
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
private val REQUIRED_PERMISSIONS_LOCATION = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)


class MainActivity : AppCompatActivity(), LifecycleOwner {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var viewFinder: TextureView
    //private val mCamera: Camera? = null
    //private val exposure = 0

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.view_finder)


        // 相机启动
        activateCameraBtn.setOnClickListener {
            if (allPermissionsGranted()) {
                startCamera()
                val location = getLocation(this)
                if (location != null) {
                    showLocation(location)
                }
                }
             else {
                ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
                )
            }
        }

        viewFinder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }

    }

    /*
    private fun setting() {
        val parameters: Camera.Parameters = mCamera.getParameters()
        //曝光度 -12 ~ 0 ~ 12
        parameters.setExposureCompensation(exposure)
        mCamera?.setParameters(parameters)
    }
     */


    private fun getLocation(context: Context): Location? {
        val locMan = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val checkCameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        val checkCallPhonePermission =
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (checkCallPhonePermission != PackageManager.PERMISSION_GRANTED || checkCameraPermission !=
            PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS_LOCATION, 2)
        }
        //var location = locMan.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        //if (location == null) { location = locMan.getLastKnownLocation(LocationManager.NETWORK_PROVIDER) }
        return locMan.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
    }

    @SuppressLint("SetTextI18n")
    private fun showLocation(location: Location){
        loc.text ="longitude：${location.longitude}, latitude：${location.latitude}"
    }


    //免安装Opencv manager
    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            println("Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            println("OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    // OpenCV库加载并初始化成功后的回调函数
    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    println("OpenCV loaded successfully")
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    private fun startCamera() {

        //预览useCase的实现
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetResolution(Size(viewFinder.width, viewFinder.height))
        }.build()

        val preview = Preview(previewConfig)

        preview.setOnPreviewOutputUpdateListener {
            val parent = viewFinder.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)
            viewFinder.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        // 图像分析useCase的实现
        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(
                ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE
            )
        }.build()

        //一个实例(instance)
        val imageAnalyzer = ImageAnalyze(applicationContext)
        imageAnalyzer.setOnAnalyzeListener(object : ImageAnalyze.OnAnalyzeListener {

            // change view outside the main thread
            override fun getDayResult(dayImage: Bitmap) {
                viewFinder.post {
                    imageGen.setImageBitmap(dayImage) // Display night2day result
                }
            }

            override fun getFPSResult(fpsShow: String) {
                viewFinder.post {
                    fps.text = fpsShow // Display current fps
                }
            }

            override fun getLoc(fakeLoc: String) {
                viewFinder.post {
                    loc.text = fakeLoc // Display current location
                }
            }

            override fun getWarning(warningText: String) {
                viewFinder.post {
                    warning.text = warningText // Display current location
                }
            }


        })
        val analyzerUseCase = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, imageAnalyzer)
        }

        // UseCase预览和图像分析
        CameraX.bindToLifecycle(this, preview, analyzerUseCase)
    }

    private fun updateTransform() {
        val matrix = Matrix()
        val centerX = viewFinder.width / 2f
        val centerY = viewFinder.height / 2f

        val rotationDegrees = when (viewFinder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)

        //textureView反映
        viewFinder.setTransform(matrix)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                viewFinder.post { startCamera() }
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    companion object{
        val TAG = MainActivity::class.java.simpleName
    }
}
