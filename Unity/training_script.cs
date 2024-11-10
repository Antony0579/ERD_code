using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Text;
using System.Globalization;
using System;
using System.Collections;
using System.Threading;
using System.IO.Ports;
using System.Linq;

public class training_script : MonoBehaviour
{
    // 1:左前，2:右前，3:左側，4:右側

    //31 最後一次留作中止指令 

    private static int Act_trials = 60; 
    private static int Length_act = Act_trials * 2; // AO 休息*2
    private int[] instruction = new int[Length_act + 1];
    private int[] waitingTime = new int[Length_act + 1];

    public GameObject signCross;
    public GameObject sign_correct;
    public GameObject left_hand;
    public GameObject right_hand;
    //public Text Point;
    private int waitSec;
    private int i = 0;
    public SerialPort sp_ard, sp_tdcs, sp_vib;

    // public string CheckFile = @"D:\\chang_unity\\new\\my_train\\check_save.txt"; //判斷是否需要開始執行鏡像學習的指令 
    //=============================================
    public GameObject human;
    Animator animator;
    private string CheckFile = @"C:/110521086_廖柏勛/0501實驗/check_save.txt";
    //private string MI_File = @"C:/110521086_廖柏勛/0501實驗/pred.txt";
    //private string RestFile_M = @"C:/110521086_廖柏勛/0501實驗/exp/";


    //public int handside;//1左手，2右手
    [Tooltip("開啟Com Port送Trigger信號到Arduino")]
    [SerializeField] private bool Action_switch = false;
    [SerializeField] private bool vibrate_switch = false;
    //public string Action_switch = "0";//啟動測試(收資料時記得歸零)
    public string arduino_COM = "";
    public string vibrate_bt_COM = "";



    //===============================================

    // 收資料用 0410
    // 修正：左手正確1，右手正確2
    // 0514 新增記分板
    // 0515 修改開關、刪除FES
    // 0519 修改 random list 
    // 0523 刪除多出的 trigger，更改輸入數字

    private bool Check = true; //判斷是否需要開始執行鏡像學習的指令 啟動invokerepeating
    private int demo_end = 0;
    private bool if_modify = false;
    private int score = 0;

    // Start is called before the first frame update

    void Start()
    {
        signCross.SetActive(false);
        left_hand.SetActive(false);
        right_hand.SetActive(false);
        sign_correct.SetActive(false);
        animator = human.GetComponent<Animator>();
        human.SetActive(true);

        score = 0;

        // 0129
        if (arduino_COM != "")
        {
            sp_ard = new SerialPort("COM" + arduino_COM.ToString(), 115200);
            sp_ard.Open();
        }
        else
        { Debug.Log("NO_arduino_COM"); }

        if (vibrate_bt_COM != "" && vibrate_switch == true)
        {
            sp_vib = new SerialPort("COM" + vibrate_bt_COM.ToString(), 9600);
            sp_vib.Open();
        }
        else
        { Debug.Log("NO_vibrate_bt_COM"); }

        setRandom();
        //save_data();
    }

    // Update is called once per frame
    void Update()
    {

        if (!Action_switch)
        {
            //在update裡 偵測到Action_switch修改後 執行一連串的指令
            if (File.ReadAllText(CheckFile) == "1")
            {
                Action_switch = true;
            }

        }


        if (Check)
        {
            //Debug.Log("NotSaving");
            if (Action_switch) // && Unity_switch == 1
            {
                Check = false;
                int resttime = waitingTime[i];

                // Debug.Log("waitingTime: " + resttime);
                Main_func();
                StartCoroutine(Delay(resttime));

            }
        }

        //在update裡 偵測到所有指令結束後 結束unity
        if (i == (instruction.Length))
        {
            Debug.Log("THE　END");

            while (if_modify == false)
            {
                if (System.IO.File.Exists(CheckFile))
                {
                    try
                    {
                        File.WriteAllText(CheckFile, demo_end.ToString());
                        if_modify = true;
                    }

                    catch
                    {
                        print("saving error");
                    }
                }
            }
            UnityEditor.EditorApplication.isPlaying = false;
        }
    }

    IEnumerator Delay(float time)
    {
        yield return new WaitForSeconds(time);
        Check = true;
    }

    private int handside_set = 0;
    void setRandom()
    {
        System.Random rnd = new System.Random();
        System.Random rnd2 = new System.Random();

        List<int> values = new List<int>();
        for (int i = 0; i < (Act_trials / 2); i++)
        {
            values.Add(1);
            values.Add(2);
        }

        //foreach (var x in values)
        //{Debug.Log(x.ToString());}
        System.Random rnd3 = new System.Random();
        var shuffled_list = values.OrderBy(_ => rnd3.Next()).ToList();

        for (int j = 0; j < Length_act; j++)
        {
            if (j % 2 == 0)  // 換方向
            {                
                handside_set = shuffled_list[j / 4];             
            }

            if (j % 2 == 0)
            {
                instruction[j] = 0;
            }
            else
            {
                instruction[j] = handside_set;//1左手，2右手 //handside
            }
        }

        for (int j = 0; j < waitingTime.Length; j++) // 設定事件所需時間
        {
            if (j % 2 == 0) //rest
            {
                waitingTime[j] = rnd2.Next(5, 7); // 休息秒數
            }            
            else //AO
            {
                waitingTime[j] = 2; // 動作秒數+1
            }
        }
    }

    void senCom(string trigger)
    {
        if (arduino_COM != "")
        {
            if (sp_ard.IsOpen)
            {
                sp_ard.Write(trigger);                
            }
        }

    }

    private string message = "VIB_ON\n";
    void senVib(string trigger)
    {
        if (vibrate_bt_COM != "" && vibrate_switch == true)
        {
            if (sp_vib.IsOpen)
            {
                message = trigger + "\n";
                sp_vib.Write(message);
                Debug.Log("Vibration, " + trigger);
            }
        }

    }

    //------------------------------------------------------------------------------------------------------------//
    void Main_func()
    {
        // print("Main_func");
        // animator.SetBool("actionR", false);

        i++;
        if (i % 2 == 0)
        {
            Debug.Log("Act "+ (i / 2).ToString());
        }
            

        if (i <= instruction.Length)
        {
            human.SetActive(true);
            //Point.text = "" + score.ToString();
            //print(Point);

            if (instruction[i - 1] == 0)
            {
                rest_set();
                senCom("0");
                //Debug.Log("Rest_end");
            }
            else if (instruction[i - 1] == 1)
            {
                //Send_FES_Com(1);
                signCross.SetActive(false);
                Invoke("coroutineLeft", 0);
                //動作時間
                if (vibrate_switch == true)
                { Invoke("coroutine_VIB_left", 0); }
                

            }
            else if (instruction[i - 1] == 2)
            {
                //Send_FES_Com(1);
                signCross.SetActive(false);
                Invoke("coroutineRight", 0);
                if (vibrate_switch == true)
                { Invoke("coroutine_VIB_right", 0); }
                
            }
        }
    }

    // Invoke 不可加參數
    private int read_MI_File = 0;
    IEnumerator wait_for_trigger_right()
    {
        Invoke("coroutineRightX", 0);
        yield return new WaitForSeconds(2);

        //Wait for 4 seconds
        float waitTime;
        if (vibrate_bt_COM != "") { waitTime = 4; }
        else { waitTime = 2; }

        float counter = 0;
        while (counter < waitTime)
        {
            //Increment Timer until counter >= waitTime
            sign_correct.SetActive(false);

            counter += Time.deltaTime;
            //Debug.Log("We have waited for: " + counter + " seconds");

            //Wait for a frame so that Unity doesn't freeze
            //Check if we want to quit this function

            //try
            //{ read_MI_File = int.Parse(File.ReadAllText(MI_File));            }
            //catch
            //{print("FAILED read MI_File ");            }

            if (true) //read_MI_File == 2
            {
                Invoke("coroutineRightO", 0);
                Invoke("coroutineRight_skip", 2);
                yield break;
            }
            //yield return null;
        }

    }

    IEnumerator wait_for_trigger_left() //平行處理
    {
        Invoke("coroutineLeftX", 0);//設定
        yield return new WaitForSeconds(2); //wait

        //Wait for 4 seconds
        float waitTime;
        if (vibrate_bt_COM != "") { waitTime = 4; }
        else { waitTime = 2; }

        float counter = 0;
        while (counter < waitTime)
        {
            //Increment Timer until counter >= waitTime
            sign_correct.SetActive(false);
            left_hand.SetActive(true);
            counter += Time.deltaTime;
            //Debug.Log("We have waited for: " + counter + " seconds");
            //Wait for a frame so that Unity doesn't freeze
            //Check if we want to quit this function

            //try
            //{ read_MI_File = int.Parse(File.ReadAllText(MI_File));            }
            //catch
            //{print("FAILED read MI_File ");            }

            if (true) //read_MI_File == 1
            {
                Invoke("coroutineLeftO", 0);
                Invoke("coroutineLeft_skip", 2);
                yield break;
            }
            //yield return null;
        }

    }

    void coroutine_VIB_left() { senVib("VIB_L"); }
    void coroutine_VIB_right() { senVib("VIB_R"); }

    void rest_set()
    {
        //sprint("---Rest---");
        signCross.SetActive(true);
        left_hand.SetActive(false);
        right_hand.SetActive(false);
        sign_correct.SetActive(false);
        animator.SetBool("actionR", false);
        animator.SetBool("actionL", false);
        senCom("0");
    }

    void coroutineLeft()
    {
        print(", Left_1");
        animator.SetBool("actionL", false);
        animator.SetBool("actionR", true);
        sign_correct.SetActive(false);
        left_hand.SetActive(true);
        senCom("1"); //249
    }
    void coroutineLeftX()
    {
        print(", LeftX_3");
        animator.SetBool("actionL", false);
        animator.SetBool("actionR", false);
        sign_correct.SetActive(false);
        left_hand.SetActive(true);
        senCom("3"); //251
    }
    void coroutineLeftO()
    {
        print(", LeftO_5");
        coroutineLeft_skip();
        senCom("5"); //253
        senVib("VIB_L");
        score += 1;
    }
    void coroutineLeft_skip()
    {
        animator.SetBool("actionL", false);
        animator.SetBool("actionR", true);
        sign_correct.SetActive(false);
        left_hand.SetActive(false);
    }




    void coroutineRight()
    {
        print(", Right_2");
        animator.SetBool("actionR", false);
        animator.SetBool("actionL", true);
        sign_correct.SetActive(false);
        right_hand.SetActive(true);
        senCom("2"); //250
    }
    void coroutineRightX()
    {
        print(", RightX_4");
        animator.SetBool("actionR", false);
        animator.SetBool("actionL", false);
        sign_correct.SetActive(false);
        right_hand.SetActive(true);
        senCom("4"); //252
    }
    void coroutineRightO()
    {
        print(", RightO_6");
        coroutineRight_skip();
        senCom("6"); //254
        senVib("VIB_R");
        score += 1;
    }
    void coroutineRight_skip()
    {
        animator.SetBool("actionR", false);
        animator.SetBool("actionL", true);
        sign_correct.SetActive(false);
        right_hand.SetActive(false);
    }


}
