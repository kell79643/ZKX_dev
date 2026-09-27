import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const files = process.argv.slice(2);
const apiPattern = "dlfft|cufft|fft|destroy|free|cleanup|finalize|shutdown|reset|DeviceReset|device reset|context";

for (const path of files) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const sheets = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 12000,
  });
  const matches = await workbook.inspect({
    kind: "match",
    searchTerm: apiPattern,
    options: { useRegex: true, maxResults: 1000 },
    maxChars: 60000,
    summary: "FFT lifecycle and runtime reset API audit",
  });
  console.log(`\n=== ${path} ===`);
  console.log("-- sheets --");
  console.log(sheets.ndjson);
  console.log("-- matches --");
  console.log(matches.ndjson);
  const lower = path.toLowerCase();
  const focus = lower.includes("driver api")
    ? { sheetId: "1.Driver API介绍", range: "A20:D28" }
    : lower.includes("runtime api")
      ? { sheetId: "runtime API支持情况", range: "A12:D18" }
      : { sheetId: "FFT", range: "A1:C17" };
  const focused = await workbook.inspect({
    kind: "table",
    ...focus,
    include: "values,formulas",
    tableMaxRows: 30,
    tableMaxCols: 6,
    maxChars: 20000,
  });
  console.log("-- focused rows --");
  console.log(focused.ndjson);
}
