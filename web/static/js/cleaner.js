function applyCleaningRules(rawText) {
    let text = rawText;

    // ============================================
    // STEP 1: UNIVERSAL START CLEANER
    // ============================================
    let startZone = text.substring(0, 600);
    let startRegex = /(?:^|\s)(?:1|E-1)[\.\-\s]+[A-Z`'""]/g;
    let m, lastStartIdx = -1;
    while ((m = startRegex.exec(startZone)) !== null) {
        lastStartIdx = m.index + m[0].length - 1;
    }
    if (lastStartIdx !== -1) {
        text = "1. " + text.substring(lastStartIdx).trim();
    } else {
        let altStart = text.search(/(?:^|\n|\s)(?:1|E-1)[\.\-\s]+/);
        if (altStart !== -1 && altStart < 2000) {
            let matchLen = text.match(/(?:^|\n|\s)(?:1|E-1)[\.\-\s]+/)[0].length;
            text = "1. " + text.substring(altStart + matchLen).trim();
        }
    }

    // ============================================
    // STEP 2: SPACED TITLE REMOVAL AFTER "1."
    // Fix: "1. Ex pEr iEn cEs 1. like to have been"
    // Fix: "1. Wo rks Is Fa It h Ex p rEs sEd ` Thank you"
    // ============================================

    // Pattern A: "1. [title garbage] 1. [real content]"
    // Real "1." aata hai toh pehle wala title section hata do
    text = text.replace(/^1\.\s+[A-Za-z][\w\s]{5,150}?\s+(?=1\.)/i, "");

    // Pattern B: "1. [title garbage] `[real content]" (backtick separator)
    text = text.replace(/^1\.\s+[A-Za-z][\w\s]{5,150}?[`\u2018\u2019]\s*/i, "1. ");

    // ============================================
    // STEP 3: INLINE PAGE HEADERS/FOOTERS
    // ============================================

    // Pattern A: "[num] English [date] [Title] IND CS[x] Rev: [y]"
    text = text.replace(/(?:\b\d+\s+)?English\s+\d{2}-\d{4}[A-Za-z]?\s+(?:(?!IND\s+CS)[\s\S]){3,200}?IND\s+CS\d*\s+Rev:\s*\d+/gi, " ");

    // Pattern B (NEW): "[num] thE sp o kEn Wo rd" — doosri line of header
    // e.g. "2 thE sp o kEn Wo rd" ya "32 T h E S p o k E n W o r d"
    text = text.replace(/\b\d+\s+[Tt]\s*h\s*[Ee]\s+[Ss]\s*p\s*[Oo]\s*[Kk]\s*[Ee]?\s*n\s+[Ww]\s*[Oo]\s*[Rr]\s*[Dd]\b/gi, " ");

    // Pattern C: Spaced Software Version header
    text = text.replace(/\b\d+\s+(?:T\s*h\s*e\s*s\s*p\s*o\s*k\s*e\s*n\s*W\s*o\s*r\s*d|English|E\s*n\s*g\s*l\s*i\s*s\s*h)[\sA-Za-z-]*?\d{2}-\d{4}[A-Za-z]?\s+S\s*o\s*f\s*t\s*w\s*a\s*r\s*e\s*V\s*e\s*r\s*S\s*i\s*o\s*n/gi, " ");

    // Pattern D: Title chunk + date + Software Version
    text = text.replace(/[^.,;!?]{5,40}?\b\d+\s+\d{2}-\d{4}[A-Za-z]?\s+S\s*o\s*f\s*t\s*w\s*a\s*r\s*e\s*V\s*e\s*r\s*s\s*i\s*o\s*n/gi, " ");

    // Pattern E (NEW): Spaced title footer with trailing number
    // e.g. "Wo rks Is Fal th Ex p rEs sEd 5" or "Ex pEr iEn cEs 3"
    // Safe check: only removes if [a-z][A-Z] pattern found = confirmed PDF artifact
    text = text.replace(/\b(?:[A-Za-z]{1,6}\s+){2,15}\d+\b/g, function(match) {
        return /[a-z][A-Z]/.test(match) ? " " : match;
    });

    // ============================================
    // STEP 4: END GARBAGE (strict — only last 2500 chars)
    // ============================================
    const endGarbageRegex = /(?:This Message by Brother|Reprinted in 20\d{2}|©\s*\d{4}\s*V\s*G\s*R|ALL RIGHTS RESERVED|V\s*o\s*i\s*c\s*E\s*O\s*f\s*G\s*o\s*d|\(812\)\s*256-1177)/i;
    let tailLength = Math.min(2500, text.length);
    let tail = text.slice(-tailLength);
    let endMatch = tail.match(endGarbageRegex);
    if (endMatch) {
        text = text.slice(0, -tailLength + endMatch.index).trim();
        text = text.replace(/\b\d+\s+(?:T\s*h\s*E\s*s\s*p\s*o\s*k\s*E\s*n\s*W\s*o\s*r\s*d|English)[\sA-Za-z-]*?\d{2}-\d{4}[A-Za-z]?\s*(?:S\s*o\s*f\s*t\s*w\s*a\s*r\s*e\s*V\s*e\s*r\s*s\s*i\s*o\s*n)?\s*$/i, "");
    }

    return text.trim();
}
