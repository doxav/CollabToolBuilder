function formatPythonDict2Table(fullText, includeHeader = false) {
    try {
        const dataArray = JSON.parse(fullText.replace(/'/g, '"')); // Parse the JSON string
        
        let tableContent = '';
        if (includeHeader) {
            tableContent += 'Value\tKey\n';
            tableContent += '------------------------------------------\n';
        }

        dataArray.forEach(dict => {
            for (const [key, value] of Object.entries(dict)) {
                let formattedValue = typeof value === 'number' && !Number.isInteger(value) ? value.
                toFixed(2) : value;
                if (Array.isArray(value)) {
                    formattedValue = `[${value.join(',')}]`
                }
                console.log(`${formattedValue}`)
                tableContent += `${formattedValue}\t${key}\n`;}
            if (includeHeader) {
                tableContent += '------------------------------------------\n';}
        });

        return tableContent;
    } catch (e) {
        return `Failed to format data. Original content:\n\n${fullText}`;
    }
}

const index = 3
const data = '0. [32mSUCCESS[0m / SCORE: [0.76,0.00,0.00,0.97,0.97,0.00,0.00] / TIME: 62.762158155441284s / CODE: [{"plan/titles similarity (top:1, worst:0)": 0.75887, "sections contents similarity (top:1, worst:0)": 0.0, "sections resources similarity (top:1, worst:0)": 0.0, "sections count (top:1, <1:too short, >1:too long)": 0.9687, "titles count (top:1, <1:too short, >1:too long)": 0.9687, "sections contents length (top:1, <1:too short, >1:too long)": 0.0, "sections contents non-empty (top:1, <1:too short, >1:too long)": 0.0, "Summary": "No paper provided for review.", "Strengths": []}]'

const data2 = '0. [32mSUCCESS[0m / SCORE: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,No paper provided for review.,[],[],0,0,0,0,[],[],False,0,0,0,0,0,Reject] / TIME: 5.66608738899231s / CODE: [{plan/titles similarity (top:1, worst:0): 0.0, sections contents similarity (top:1, worst:0): 0.0, sections resources similarity (top:1, worst:0): 0.0, sections count (top:1, <1:too short, >1:too long): 0.0, titles count (top:1, <1:too short, >1:too long): 0.0, sections contents length (top:1, <1:too short, >1:too long): 0.0, sections contents non-empty (top:1, <1:too short, >1:too long): 0.0, Summary: No paper provided for review., Strengths: [], Weaknesses: [], Originality: 0, Quality: 0, Clarity: 0, Significance: 0, Questions: [], Limitations: [], Ethical Concerns: false, Soundness: 0, Presentation: 0, Contribution: 0, Overall: 0, Confidence: 0, Decision: Reject}]'
// Regex patterns
const statusRegex = /\[\d+m(SUCCESS|FAILED)\[\d+m/;
const scoreRegex = /SCORE:\s+(\[.*?\])/;
const timeRegex = /TIME:\s+([\d.]+)s/;
const codeRegex = /CODE:\s+(\[.*\])$/;

// Extract values
const statusMatch = data.match(statusRegex);
const scoreMatch = data.match(scoreRegex);
const timeMatch = data.match(timeRegex);
const codeMatch = data.match(codeRegex);

if (statusMatch) {
    console.log("Match found:", statusMatch);
    const score = statusMatch[1];
    const timeT = parseFloat(timeMatch[1]).toFixed(1);
    let textContent = score;
    if (score === "SUCCESS") {
        console.log(`button.style.backgroundColor = "green"`);
    } else {
        console.log(`button.style.backgroundColor = "red"`);
    }

    if (scoreMatch) {
        const detailScores = formatPythonDict2Table(codeMatch[1]);
        console.log(`Test score of solution ${index} in ${timeT}s:\n${detailScores}`);
    }
} else {
    console.log(`button.textContent = "Scoring failed"`);
    // button.style.backgroundColor = "red";
}