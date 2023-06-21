# eKYC
[機敏資料](https://whimsical.com/ekyc-FDwjAQ2ALRKA1bRC5pNGVh#)
*  **每個Stage都須討論**
    *  input/output(request/response) - Json
    *  操作流程（使用者畫面、Timer、有無按鈕、表格、顯示資訊(OCR)）
    *  controller 回傳加入 Messsage(說明)print
    *  
## 逐項討論
###  **Stage 0 - 開始進行認證:**
![](https://i.imgur.com/htv4Orb.png)

### **Stage 1 - 輸入身分資訊:**
*  前端(待補畫面)
    *  使用者畫面:
        * 三個欄位輸入
            * Name
            * Date
            * ID Number
    *  回傳:
    ```
    Json:
    { 
        Session_ID:
        Stage:
        Identity_Info:
            Name:
            Date:
            ID:
    }
    ```
        Identity_Info:
            使用者輸入之身份資訊
                    
            
* 後端
    * 回傳:
    ```
    Json:
    {
        Session_ID:
        Current_Stage:
        Next_Stage:
        Message:
    }
    ```
        
###  **Stage 2 - 身分證正面擷取:**
![](https://i.imgur.com/KTDmJDA.png)

*  前端
    *  使用者畫面:
        1. 有無綠框? (O)
        2. 按下按鈕後，倒數3、2、1
        3. 回傳圖片是"**傳送整張畫面影像**"
        4. 提示"請把證件放在綠框內"
    *  回傳:
    ```
    Json:
    {
        Session_ID:
        Stage:
        Img:
    }
    ```
        Img:
            身分證正面照
*  後端
    *  回傳:
    ```
    Json:
    {
        Session_ID:
        Current_Stage:
        Next_Stage:
        Message:
    }
    ```
### **Stage 3 - 身分證旋轉過程:**
開始擷取影像為 - 身分證放入綠框內後，開始計時三秒(往上)
---上下翻轉---
1.正 -> 上梯形
2.上梯形 -> 正
3.正 -> 下梯形
4.下梯形 -> 正

**截圖回傳正面最後一幀**

---左右翻轉---
5.正 -> 左梯形
6.左梯形 -> 正
7.正 -> 右梯形
8.右梯形 -> 正

#### **Stage 3.1 -上下翻轉**
*  前端
    *  使用者畫面:
        1. 以梯形方式呈現上下綠色框
        2. 回傳圖片是"**傳送整張畫面影像**"，而非單獨梯形影像
        (O) ![](https://i.imgur.com/2MqvXZE.png) 
        (X) ![](https://i.imgur.com/EP2UGIn.png)
        
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Stage:
        Img:
        id_flag:False
    }
    ```
        Img:
            身分證旋轉中影片拆幀
            每1/10秒取一幀往後端送
        id_flag:
            照片旋轉過程綠框為正面長方形：True
            若為旋轉過程：False
#### **Stage 3.2 - 上下翻轉後接左右翻轉的正面畫面**
![](https://i.imgur.com/fnnsnIM.png)
* 前端
    * 使用者畫面:
        * **不須特別告知使用者停留，以轉場方式接續左右翻轉，我們僅取影像。**
        * 回傳：
        ```
        Json:
        {
            Session_ID:
            Stage:
            Img:
            id_flag:True(這裡不一樣)
        }
        ```
            Img:
                身分證旋轉中影片拆幀
                每1/10秒取一幀往後端送
            id_flag:
                照片旋轉過程綠框為正面長方形：True
                若為旋轉過程：False

#### **Stage 3.3 - 左右翻轉**
* 前端
    * 使用者畫面:
        * 同上下翻轉
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Stage:
        Img:
        id_flag:False
    }
    ```
        Img:
            身分證旋轉中影片拆幀
            每1/10秒取一幀往後端送
        id_flag:
            照片旋轉過程綠框為正面長方形：True
            若為旋轉過程：False
*  後端（**結束後**）
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Current_Stage:
        Next_Stage:
        Message:
    }
    ```
### **Stage 4 - 身分證正面OCR:**
*  前端
    * 使用者畫面:
        1. 有無綠框(O)
        2. 有無呈現使用者資訊(O)-message(講好怎麼分行)
        3. 有無讓使用者使用按鈕(O)，有的話前端是回傳使用者按按鈕當下的影像?
        4. Counter設定三次
    * 回傳：

    ```
    Json:
    {
        Session_ID:
        Stage:
        Img:
    }
    ```
*  後端
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Current_Stage:
        Next_Stage:
        Message:
        Correct:
    }
    ```
        Message:
            render這個message
        Correct:
            OCR判斷正確或否
###  **Stage 5 - 身分證反面:**
*  前端
    * 使用者畫面:
        1. 有無綠框(O)
        2. 有無呈現使用者資訊(O)，但
        3. 有無讓使用者使用按鈕(O/X)，有的話前端是回傳使用者按按鈕當下的影像?
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Stage:
        Img:
    }
    ```

*  後端
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Current_Stage:
        Next_Stage:
        Message:
        Correct:
    }
    ```
### **Stage 6 - FV&PAD:**
![](https://i.imgur.com/xGapKkD.png)

*  前端
    * 使用者畫面:
        1. 同身分證，畫出人頭框線，回傳**整張圖片**
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Stage:
        Img:
    }
    ```
*  後端
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Current_Stage:
        Next_Stage:
        Message:
    }
    ```
### **Stage 7 - 驗證結果FSE:**
![](https://i.imgur.com/GzKC6UI.png)

*  前端
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Stage:
    }
    ```
*  後端
    * 回傳：
    ```
    Json:
    {
        Session_ID:
        Score:
        Img1:
        Img2:
        Message:
    }
    ```
        Img1:
            身分證正面
        Img2:
            身分證反面
        Score：
            最終驗證結果成績[0,1]