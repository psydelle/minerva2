<?php
// CHANGE OR REMOVE BEFORE SHARING //
$home = '/home/wwwsydsphdresear';
$data_upload_dir = "$home/data_submissions/minerva2";

// read secrets from /home/wwwsydsphdresear/env_secrets.json
$secrets = json_decode(file_get_contents("$home/env_secrets.json"), true);
$PROLIFIC_API_TOKEN = $secrets['PROLIFIC_API_TOKEN'];
/////////////////////////////////////


$json = file_get_contents('php://input');
$obj = json_decode($json, true);
// error_log("obj: ".print_r($obj, true));
$prolific_id = $obj["prolific_id"];
$prolific_study_id = $obj["prolific_study_id"];
$fold_id = strval($obj["folda"])."_".strval($obj["foldb"]);

function check_id_has_started($prolific_id, $prolific_study_id, $prolific_api_token, $url = null) {
    $ch = curl_init();
    if (is_null($url)) {
        // top recursion level, so set the url to the study submissions endpoint
        curl_setopt($ch, CURLOPT_URL, "https://api.prolific.com/api/v1/studies/$prolific_study_id/submissions/");
    }
    else {
        // not top recursion level, so use the url passed in
        curl_setopt($ch, CURLOPT_URL, $url);
    }
    // configure curl
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, array("Content-Type: application/json", "Authorization: Token $prolific_api_token"));

    // execute curl
    $response = curl_exec($ch);

    // // DEBUG: print curl config to error log
    error_log('info'.print_r(curl_getinfo($ch), true));

    if(curl_error($ch)) {
        error_log("curl error: ".curl_error($ch));
    } 
    curl_close($ch);

    $response = json_decode($response, true);
    // // DEBUG: print response to error log
    error_log("response: ".print_r($response, true));
    $participants = $response["results"];
    error_log("participants: ".print_r($participants, true));
    foreach ($participants as $participant) {
        // if prolific_id is in the list of participants, return true
        if ($participant["participant_id"] == $prolific_id) {
            return true;
        }
    }
    if (!is_null($response['_links']["next"]['href'])) {
        // if there is a next page, recurse
        return check_id_has_started($prolific_id, $prolific_study_id, $prolific_api_token, $response['_links']["next"]["href"]);
    }
    // if prolific_id is not in the list of participants and there is no next page, return false
    return false;
}

$path = $data_upload_dir."/".$fold_id."_".$prolific_id.".json"; 


// check prolific_id is not empty
if (empty($prolific_id)) {
    error_log("empty prolific_id: ".$prolific_id);
}
// check prolific_id is alphanumeric
elseif (!ctype_alnum($prolific_id)) {
    error_log("bad prolific_id: ".$prolific_id);
}
// check prolific_id is between 1 and 100 characters
elseif (strlen($prolific_id) < 1 || strlen($prolific_id) > 100) {
    error_log("incorrect length of prolific_id: ".$prolific_id);
}

elseif (substr(realpath(dirname($path)), 0, strlen($data_upload_dir))!=$data_upload_dir) {
    error_log("attempt to write to bad path: ".$path);
} 

// elseif (!check_id_has_started($prolific_id, $prolific_study_id, $PROLIFIC_API_TOKEN)) {
//     error_log("prolific_id has not started the study: ".$prolific_id);
// }

else {
    $outfile = fopen($path, "w");
    $jsonString = json_encode($obj, JSON_PRETTY_PRINT);
    fwrite(
        $outfile,
        $jsonString
    );
    fclose($outfile);
}
?>