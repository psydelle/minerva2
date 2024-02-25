/******************************************************************************/
/*** Initialise jspsych *******************************************************/
/******************************************************************************/

var jsPsych = initJsPsych({
  // on_finish: function () {
  //   jsPsych.data.displayData("json");
  //   var data = jsPsych.data.get().json();
  on_finish: function (data) {
    document.body.innerHTML = "<img src='checkmark.gif' alt='green check mark gif' style='width:300px;height:300px; margin='auto'>\
    <h2 style='text-align:center'>Data were saved. <br> You may now safely close your browser window.</h2>";
    //setTimeout(function () { location.href = "https://app.prolific.co/submissions/complete?cc="+ PROLIFIC_COMPLETION_CODE}, 5000); 
  },
  // send post request to /save_data with the participant's data
  // save_data(data);

  exclusions: {
    min_width: 800,
    min_height: 600
  },
  // on_trial_finish: function(data){
  //   var trial_response = JSON.stringify(data.response)
  //   data.response = trial_response
  //       // send post request to /save_data with the participant's data
  //       fetch("/save_data", {
  //         method: "POST",
  //         body: JSON.stringify(data),
  //         headers: {
  //           "Content-Type": "application/json",
  //         },
  //       });

  // }
});

window.mobileAndTabletCheck = function () {
  let check = false;
  (function (a) { if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4))) check = true; })(navigator.userAgent || navigator.vendor || window.opera);
  return check;
};

// exclude mobile browsers
if (window.mobileAndTabletCheck()) {
  alert("This study is not compatible with your browser. Please switch to a desktop computer to complete this task.");
  document.body.innerHTML = "<h1>This study is not compatible with your browser. Please switch to a desktop computer to complete this task.</h1>";
};

// write a function to call fold and prolific id from the url

function getFold() {
  let fold_idx = Number(jsPsych.data.getURLVariable("fold"));
  if (fold_idx == null) {
    console.error("fold not specified in URL");
    fold_idx = -1;
  }
  return fold_idx;
}

function getProlificID() {
  let prolific_id = jsPsych.data.getURLVariable("PROLIFIC_PID");
  // let prolific_id = `drnlppilot${jsPsych.randomization.randomID(10)}`;
  return prolific_id;
}

function getProlificStudyID() {
  return jsPsych.data.getURLVariable("PROLIFIC_STUDY_ID");
}

console.log(STIMULI);
// let fold_idx = getFold();
//let getProlificID = jsPsych.data.getURLVariable("PROLIFIC_PID")
console.log(getFold(), getProlificID());

// write a function to mark a trial as real or not that i can call in on_finish

function ajtOnFinish(data) {
   data.dataType = "ajtTrial";
   data.condition = jsPsych.timelineVariable("type");
   data.goldResponse = jsPsych.timelineVariable("correct_ajt_response");
};

function markTrialAsFake(data) {
  data.dataType = "discard";
}

function markTrialAsInfo(data) {
  data.dataType = "info";
}


jsPsych.randomization.setSeed(getFold())

// Filter the stimuli array by fold to get the stimuli for the current participant

let filtered = STIMULI.filter(obj => obj.fold === getFold());

console.log("stimuli set for fold", filtered);

if (filtered.length != 20) {
  console.error("not 20 items in set!"); 
}

// check if there are 10 collocations and 10 productive combinations in the set

if (filtered.filter(obj => obj.type === "collocation").length != 10) {
  console.error("not 10 collocations in set!"); 
}

if (filtered.filter(obj => obj.type === "productive").length != 10) {
  console.error("not 10 productive combinations in set!"); 
}

let SUB_STIMULI = filtered.concat(BASELINES);

if (SUB_STIMULI.length != 40) {
  console.error("not 40 items in participant set!"); 
}

jsPsych.randomization.setSeed(getProlificID()) 

SUB_STIMULI = jsPsych.randomization.shuffle(SUB_STIMULI)

// ensure first stimulus is productive
while (SUB_STIMULI[0].type !== "productive") {
  SUB_STIMULI = jsPsych.randomization.shuffle(SUB_STIMULI)
}

console.log(SUB_STIMULI);

let literalnessStim = jsPsych.randomization.shuffle(filtered);

/*exclude mobile browsers*/

var browserCheckTrial = {
  type: jsPsychBrowserCheck,
  inclusion_function: (data) => {
    return data.mobile === false;
  },
  exclusion_message: (data) => {
    if (data.mobile) {
      return "<h3>You must use a desktop/laptop computer with a keyboard and a mouse to participate in this study.</h3>";
    }
  },
  on_finish: markTrialAsFake,
};


// function to check if a participant has given consent to participate.
var check_consent = function (elem) {
  if (document.getElementById('consent_checkbox').checked) {
    return true;
  }
  else {
    alert("If you wish to participate, you must check the box next to the statement 'I agree to participate in this study.'");
    return false;
  }
  return false;
};


/******************************************************************************/
/*** Socio-demographics *******************************************************/
/******************************************************************************/
var sociodemo1 = {
  type: jsPsychSurveyText,
  preamble:
    "<h3 style='text-align:center'><b>Please answer the following questions.</b>\
    </h3>",
  questions: [
    {
      prompt: "How old are you?",
      placeholder: "Enter your age in numbers (e.g., 18)",
      required: true,
      name: "Age",
    },
  ],
  on_finish: markTrialAsInfo,
};

var sociodemo2 = {
  type: jsPsychSurveyMultiChoice,
  preamble:
    "<h3 style='text-align:center'><b>Please answer the following questions.</b>\
  </h3>",
  questions: [
    {
      prompt: "Handedness:",
      name: "Handedness",
      options: ["Left", "Right", "Ambidextrous"],
      required: true,
    },
    {
      prompt: "Vision:",
      name: "Vision",
      options: [
        "Normal",
        "Corrected (I wear contacts/glasses.)",
        "Other (I have other vision problems.)",
      ],
      required: true,
    },
    {
      prompt:
        "Have you been diagnosed with a language disorder (e.g., dyslexia, SLI, etc.)?",
      name: "LanguagePathology",
      options: ["Yes", "No"],
      required: true,
    },
    {
      prompt: "What is your first (native) language?",
      name: "L1",
      options: ["English", "Portuguese", "Spanish", "Mandarin", "French", "German", "Hindi", "Other"],
      required: true,
    },
  ],
  on_finish: markTrialAsInfo,
};


var pilot_comments = {
  type: jsPsychSurveyText,
  preamble:
    "<h1 style='text-align:center'><b>Feedback</b>\
    </h1>",
  questions: [
    {
      prompt: "I'd really appreciate your feedback! What did you think about the experiment?<br><em>(Any glitches? Confusing instructions?)</em>",
      placeholder: "",
      required: false,
      name: "PilotComments",
      rows: 10,
    },
  ],
  on_finish: markTrialAsInfo,
};

/******************************************************************************/
/*** Judgment trials **********************************************************/
/******************************************************************************/
// Define the fixation trial
var fixation = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p style="font-size:70px;"> + </p>',
  trial_duration: 350,
  response_ends_trial: false,
  on_finish: markTrialAsFake,
};

// Define the countdown trials
var countdown1 = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p style="font-size:70px; font-weight:bolder;"> 1 </p>',
  trial_duration: 850,
  response_ends_trial: false,
  on_finish: markTrialAsFake,
};

var countdown2 = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p style="font-size:70px; font-weight:bolder;"> 2 </p>',
  trial_duration: 850,
  response_ends_trial: false,
  on_finish: markTrialAsFake,
};

var countdown3 = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: '<p style="font-size:70px; font-weight:bolder;"> 3 </p>',
  trial_duration: 850,
  response_ends_trial: false,
  on_finish: markTrialAsFake,
};

var countdown_begin = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: "<p style='font-size:70px; font-weight:bolder;'> Let's go! </p>",
  trial_duration: 850,
  response_ends_trial: false,
  on_finish: markTrialAsFake,
};

// stitch together countdown trials
var countdown = {
  timeline: [countdown3, countdown2, countdown1, countdown_begin],
  randomize_order: false,
};



// Acceptability judgment trials

var ajt_practice_trial = {
  type: jsPsychCategorizeHtml,
  key_answer: jsPsych.timelineVariable("correct_ajt_response"),
  text_answer: "letter",
  choices: ["y", "n"],
  trial_duration: 8000,
  feedback_duration: 3000,
  correct_text: "<h1 class='prompt'> &#128515 </h1>\
  <h3 class='prompt' style='color: green;'> That's right! </h3>",
  incorrect_text: "<h1 class='prompt''> &#128534 </h1>\
  <h3 class='prompt' style='color: red;'> Hmmm... </h3>",
  prompt:
    `<p id="prompt"><em>Would this word combination be used by a native English speaker?\
     <br> Press <b>Y</b> for yes or <b>N</b> for no.</em></p>`,
  stimulus: function () {
    var stim =
      '<p id="stimuli"">' +
      jsPsych.timelineVariable("item") +
      "</p>";
    return stim;
  },
  on_finish: markTrialAsFake,
};

var ajt_test_trial = {
  type: jsPsychHtmlKeyboardResponse,
  prompt:
    `<p id="prompt"><em>Would this word combination be used by a native speaker of English? <br> Press <b>Y</b> for yes or <b>N</b> for no. </em></p>`,
  choices: ["y", "n"],
  trial_duration: 8000,
  on_finish: ajtOnFinish,
  stimulus: function () {
    var stim =
      '<p id="stimuli">' +
      jsPsych.timelineVariable("verb") + " " + jsPsych.timelineVariable("noun_pl") +
      "</p>";
    return stim;
  },
};

// Stitch the two together - use the items variable for the timeline_variables
var ajt_practice = {
  timeline: [fixation, ajt_practice_trial],
  timeline_variables: AJT_PRACTICE_STIMULI,
  randomize_order: true,
};

var ajt_test = {
  timeline: [fixation, ajt_test_trial],
  timeline_variables: SUB_STIMULI,
  randomize_order: false,
};




/******************************************************************************/
/*** Instruction trials *******************************************************/
/******************************************************************************/

// jspsych make the prompt appear above the stimulus

// declare the block.

//add a block to check consent and then run the experiment

var participant_information = {
  type: jsPsychExternalHtml,
  url: "participant_info.html",
  cont_btn: "start",
  check_fn: check_consent,
  on_finish: markTrialAsFake,
};

var welcome = {
  type: jsPsychHtmlButtonResponse,
  stimulus:
    "<h1>Welcome!</h1>\
    <br><h3>Thank you for agreeing to participate in this study.</h3>\
    <p style='text-align:left'>You will be given a short, engaging, and simple task to perform.\
    There will be instructions &#128064 and practice with feedback. Please pay close attention to them.\
    Before you begin, please ensure that you are in a quiet space where you are unlikely to be distracted or interrupted.\
    <mark>Keep in mind that there are no right or wrong answers.</mark> We are only interested in your intuitions as a native English speaker.\
    So, relax and have fun. You should be done in about 5 minutes.</p>\
    <h3>Click the button below to proceed.</h3>",
  choices: ["Click to proceed"],
  button_html: '<button>%choice%</button>',
  on_finish: markTrialAsFake,
};

var ajt_instructions = {
  type: jsPsychInstructions,
  pages: [
    "<h1>&#128064 Instructions</h1> <br> <h4>[Use arrow keys to navigate.]</h4>",
    "<p style='text-align:left'>In this task, you will read word combinations in English without context. You have to determine if they sound \
  acceptable to you. By acceptable, we mean whether you think a native English speaker would use this word combination in a conversation.\
  In other words, do you think it would sound odd for someone to say this to you, as if they don't speak English natively?</p>",
    "<h2>&#128064 Instructions</h2>\
  <p style='text-align:left'> There are no right or wrong answers. We are interested in your initial intuitions about these word combinations.\
  Please reply as quickly and accurately as possible by pressing the <b>Y</b> key for <em>yes</em> or the <b>N</b> key for <em>no</em>.\
  <br>If you do not respond within 8 seconds, you will automatically be moved on to the next word combination.</p>",
    "<h2>&#128064 Instructions</h2>\
  <p style='text-align:left'> You will first be given a practice round with feedback before the actual task begins.\
  Pay close attention to the feedback as it will help you understand the task better.",
  ],
  allow_keys: true,
  on_finish: markTrialAsFake,
};

var ajt_begin_practice = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus:
    "<h2>Fingers at the ready! </h2>\
  <img src='https://www.sydsphdresearch.ppls.ed.ac.uk/ProjectLiteral/experiment/yn-keys.gif' alt='keys' style='width:300px;height:300px;'>\
  <p style='text-align:center'> Keep your <b>index fingers</b> over the <b>Y</b> (<em>yes</em>) and <b>N</b> (<em>no</em>) keys.\
   Please refrain from doing this task one-handedly.\
  If you don't respond within 8 seconds you will automatically be moved on to the next combination.</p>\
  <h4> Remember, we are interested in your first impressions!</h4>",
  choices: ["Enter"],
  prompt: ["<h3 style='text-align:center'> <br> Press 'Enter' to begin.</h3>"],
  on_finish: markTrialAsFake,
};

var ajt_end_practice = {
  type: jsPsychHtmlButtonResponse,
  stimulus:
    "<h1 style='font-size:700%'> &#128077 </h1>\
    <br><h2>Well done!</h2>\
    <h4>That was the practice round. Let's move on to the test round.</h4><br>",
  choices: ["Click to proceed"],
  button_html: '<button>%choice%</button>',
  on_finish: markTrialAsFake,
};

var ajt_begin_test = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus:
    "<h2>We're now starting the test round! </h2>\
  <img src='https://www.sydsphdresearch.ppls.ed.ac.uk/ProjectLiteral/experiment/yn-keys.gif' alt='keys' style='width:300px;height:300px;'>\
  <p style='text-align:center'> Keep your <b>index fingers</b> over the <b>Y</b> (<em>yes</em>) and <b>N</b> (<em>no</em>) keys.\
  Please refrain from doing this task one-handedly.\
  If you don't respond within 8 seconds you will automatically be moved on to the next combination.</p>\
  <h4> Remember, we are interested in your first impressions!</h4>",
  choices: ["Enter"],
  prompt: ["<h3 style='text-align:center'> <br> Press 'Enter' to begin.</h3>"],
  on_finish: markTrialAsFake,
};

var ajt_end = {
  type: jsPsychHtmlButtonResponse,
  stimulus:
    "<img src='https://www.sydsphdresearch.ppls.ed.ac.uk/ProjectLiteral/experiment/task-complete.gif' alt='party popper gif' height='300px' width='300px'>\
  <br><h2>Great job! You've completed Task 1.</h2>\
  <h4>Now let's move on to the next task.</h4>",
  choices: ["Click to proceed"],
  button_html: '<button>%choice%</button>',
  on_finish: markTrialAsFake,
};


var save_data_trial = {
  type: jsPsychCallFunction,
  async: true,
  func: function (done) {
    var url = 'save_data.php';
    var data_in = jsPsych.data.get();
    console.log(data_in);
    // var data_to_send = { getProlificID: jsPsych.data.getURLVariable("PROLIFIC_PID"), filedata: data_in };
    var data_to_send = { prolific_study_id: getProlificStudyID(), prolific_id: getProlificID(), fold: getFold(), filedata: data_in };
    let response = fetch(url, {
      method: 'POST',
      body: JSON.stringify(data_to_send),
      headers: new Headers({
        'Content-Type': 'application/json'
      })
    }).then(function (response) {
      if (response.ok) {
        console.log("Data saved!")
        done(response);
      } else {
        throw new Error('Network response was not ok.');
      }
    }).catch(function (error) {
      console.log(error);
      done(error);
    });
  },
  on_finish: markTrialAsFake,
};

var final_screen = {
  type: jsPsychHtmlButtonResponse,
  stimulus:
    "<img src='https://www.sydsphdresearch.ppls.ed.ac.uk/ProjectLiteral/experiment/UoELogoStacked.png' alt='Edinburgh University Logo' height='90'>\
  <img src='https://www.sydsphdresearch.ppls.ed.ac.uk/ProjectLiteral/experiment/done.gif' alt='smiley face celebrating all done gif' height='300px' width='300px'>\
  <p style='text-align:center'>Thank you for participating in our study. We hope you had fun.</p>\
  <p style='text-align:center'>Here is the Completion Code <b>CSM761KM</b>.</p>\
  <p style='text-align:center'>Please copy and paste this into Prolific to claim your payment. </p>",
  choices: ["Click to Finish"],
  button_html: '<button>%choice%</button>',
  on_finish: markTrialAsFake,
};


/******************************************************************************/
/*** Build the timeline *******************************************************/
/******************************************************************************/

var full_timeline = [
  browserCheckTrial,
  participant_information,
  welcome,
  ajt_instructions,
  ajt_begin_practice,
  countdown,
  ajt_practice,
  ajt_end_practice,
  ajt_begin_test,
  countdown,
  ajt_test,
  ajt_end,
  sociodemo1,
  sociodemo2,
  pilot_comments,
  save_data_trial,
  final_screen,
];

/*
Call jsPsych.run to run timeline.
*/

jsPsych.run(full_timeline);

