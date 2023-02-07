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
  -  <a href="#-characters-recognition">  Characters Recognition </a>
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

## <a id="-characters-recognition"></a> Characters Recognition
### How convert characters images to text by [OCR By Machine Learning]
> this module is itrative and run foreach character
<br>


1. ####  **read the character image**                            
  <div align="center">
<img src="./readmeImages/12.png" width="200px" height="200px" alt="License plate recognition" align="center""/>
</div>
<br>
<hr/>
                                                                                                             
                                                                                                             
## <a id="contributors"></a> Contributing

> Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.


#### Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/AbdelrahmanAshrafMohamedelsayed"><img src="https://avatars.githubusercontent.com/u/97232730?s=400" width="150px;" alt=""/><br /><sub><b>Abdelrahman Ashraf</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/abdalhamedemad"><img src="https://avatars.githubusercontent.com/u/76442606?v=4" width="150px;" alt=""/><br /><sub><b>Abdelhameed Emad</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Karim-Mohamed20"><img src="https://avatars.githubusercontent.com/u/101477261?v=4" width="150px;" alt=""/><br /><sub><b>Kareem Mohamed</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/khaled-farahat"><img src="https://cdn1.img.sputnikarabic.ae/img/101897/08/1018970827_7:0:2992:1688_600x0_80_0_0_03dcfd0856edcfa44aa1892e188c1a05.jpg.webp" width="150px;"  height="150px;" alt=""/><br /><sub><b>khaled Farahat</b></sub></a><br /></td>
  </tr>
 </table>

## 🔒 License <a id ="license"></a>

> **Note**: This software is licensed under MIT License, See [License](https://github.com/AbdelrahmanAshrafMohamedelsayed/License-Plate-Recognition/blob/main/LICENSE) for more information ©Mr.Abdo.
