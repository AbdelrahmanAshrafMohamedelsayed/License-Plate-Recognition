# <img src="https://ozekirobot.com/attachments/6144/Lead_image.png" alt="License plate recognition" align="center" style="height: 75px; width:75px;"/> License plate recognition
<div align="center">
<img src="./readmeImages/logo.png" width="525px" alt="License plate recognition" align="center""/>
</div>
<br>

                                                                                                
## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents

- <a href ="#about"> 📙 About</a>
- <a href ="#Diagram"> :bar_chart: Diagram</a>
- <a href ="#started"> 💻 Get Started</a>
- <a href ="#-image-processing-algorithms-applied"> :mag_right: Image processing Algorithms applied</a>
- <a href ="#modules"> 🌎 Modules</a>
  -  <a href="#-dedect-characters">  Dedect characters </a>
  -  <a href="#Characters-Recognition">  Characters Recognition </a>
- <a href ="#contributors"> ✨ Contributors</a>
- <a href ="#license"> 🔒 License</a>
<hr/>
                     
                     
## <a id="about"></a> About
  <p>
  Our project is <strong>License plate recognition</strong>: A gate is open for specific cars based on their plate number, by capturing images of the car and transform the optical data into digital information [characters] and take a decision.
 </p>

<hr/>

<h2 id="Diagram">Diagram<h2>
<img src="./readmeImages/diagram.png" />
<hr/>
                                     
## <a id = "started"></a> <img align= center width=50px height=50px src="https://c.tenor.com/HgX89Yku5V4AAAAi/to-the-moon.gif"> Get Started 
<ol>
<li>Clone the repository

<br>

```
$ git clone https://github.com/AbdelrahmanAshrafMohamedelsayed/License-Plate-Recognition.git
```

</li>                                                                                                                                       
</ol>
<hr/>     

## <a id="Algorithms"></a> Image processing Algorithms applied
<ul>
    <li>Thresholding</li>
    <li>Segmentation</li>
    <li>Smoothing</li>
    <li>Histogram of gradient (HoG)</li>
    <li>Support Vector Machine (SVM)</li>
    <li>Smoothing</li>
    <li>
        <div>
            <h4>Morphological operations</h4>
            <ul>
                <li>Dilation</li>
                <li>Erosion</li>
                <li>Opening</li>
                <li>Closing</li>
            </ul>
        </div>
    </li>
    <li>Template matching</li>
</ul>
<hr/>  
                     
                     
## <a id="Dedect-characters"></a> Dedect characters
### How dedect the characters from the car
1. ####  **read the image**                            
  <div align="center">
<img src="./readmeImages/CAR.png" width="525px" alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>


2. ####  **Apply image preprocessing**
   1. #####  **convert image to greyscale**                            
   2. #####  **threshold image**                            
   3. #####  **erosion**                            
  <div align="center">
<img src="./readmeImages/PRE.png"  alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>
                                                                                  
                                                                                  
3. ####  **Apply canny edge detector**                            
  <div align="center">
<img src="./readmeImages/edged.png" width="525px" alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>

                                                                                  
                                                                                  
4. ####  **finding and sort the contours according to their area**                            
  <div align="center">
<img src="./readmeImages/CONTOUR.png" width="525px" alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>
                                                                                                   
5. ####  **find the plate**                            
  <div align="center">
<img src="./readmeImages/PLATE.png" width="525px" alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>


6. ####  **extract characters**                            


<table>
<tr>
<td>
<img src="./readmeImages/1.png" height=200>
</td>
  
  <td>

<img src="./readmeImages/2.png" height=200>

</td>
  <td>

<img src="./readmeImages/3.png" height=200>

</td>
  <td>

<img src="./readmeImages/4.png" height=200>

</td>
<td>

<img src="./readmeImages/5.png" height=200>

</td>
<td>

<img src="./readmeImages/6.png" height=200>

</td>
  <td>

<img src="./readmeImages/7.png" height=200>

</td>
</tr>
</table>
<br>
<hr/>

## <a id="Dedect-characters"></a> Characters Recognition
### How convert characters images to text by [OCR By Machine Learning]
> this module is itrative and run foreach character
<br>
1. ####  **read the character image**                            
  <div align="center">
<img src="./readmeImages/12.png" width="525px" alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>
