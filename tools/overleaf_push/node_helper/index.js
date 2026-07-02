#!/usr/bin/env node

const io = require("socket.io-client");

const originalHandshake = io.Socket.prototype.handshake;
io.Socket.prototype.handshake = function (fn) {
  const self = this;
  const options = this.options;
  const extraHeaders = options.extraHeaders || {};

  function complete(data) {
    if (data instanceof Error) {
      self.connecting = false;
      self.onError(data.message);
    } else {
      fn.apply(null, data.split(':'));
    }
  }

  const url = [
    'http' + (options.secure ? 's' : '') + ':/',
    options.host + ':' + options.port,
    options.resource,
    io.protocol,
    io.util.query(this.options.query, 't=' + +new Date())
  ].join('/');

  const xhr = io.util.request();
  xhr.open('GET', url, true);
  if (xhr.setDisableHeaderCheck) {
    xhr.setDisableHeaderCheck(true);
  }
  if (xhr.setRequestHeader) {
    Object.entries(extraHeaders).forEach(([key, value]) => {
      xhr.setRequestHeader(key, value);
    });
  }
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4) {
      xhr.onreadystatechange = function () {};
      if (xhr.status == 200) {
        complete(xhr.responseText);
      } else if (xhr.status == 403) {
        self.onError(xhr.responseText);
      } else {
        self.connecting = false;
        !self.reconnecting && self.onError(xhr.responseText);
      }
    }
  };
  xhr.send(null);
};

const originalXhrRequest = io.Transport.XHR.prototype.request;
io.Transport.XHR.prototype.request = function (method) {
  const req = originalXhrRequest.call(this, method);
  const extraHeaders = (this.socket && this.socket.options && this.socket.options.extraHeaders) || {};
  if (req.setDisableHeaderCheck) {
    req.setDisableHeaderCheck(true);
  }
  if (req.setRequestHeader) {
    Object.entries(extraHeaders).forEach(([key, value]) => {
      req.setRequestHeader(key, value);
    });
  }
  return req;
};

function fail(message) {
  process.stderr.write(String(message) + "\n");
  process.exit(1);
}

function connect(origin, cookie, projectId) {
  const url = `${origin}?projectId=${encodeURIComponent(projectId)}&t=${Date.now()}`;
  return io.connect(url, {
    reconnect: false,
    "force new connection": true,
    transports: ["xhr-polling"],
    extraHeaders: {
      Origin: origin,
      Cookie: cookie,
    },
  });
}

function waitForConnect(socket) {
  return new Promise((resolve, reject) => {
    socket.on("connect", resolve);
    socket.on("connect_error", reject);
    socket.on("connect_failed", reject);
    socket.on("error", reject);
    socket.on("connectionRejected", (err) => {
      reject(new Error(err && err.message ? err.message : "connectionRejected"));
    });
  });
}

function emit(socket, event, ...args) {
  return new Promise((resolve, reject) => {
    socket.emit(event, ...args, (err, ...data) => {
      if (err) {
        reject(err);
        return;
      }
      resolve(data);
    });
  });
}

async function main() {
  const [command, origin, cookie, payload] = process.argv.slice(2);
  if (!command || !origin || !cookie) {
    fail("Usage: index.js <join-project|join-doc> <origin> <cookie> <payload>");
  }

  const socket = connect(origin, cookie, payload);
  await waitForConnect(socket);

  try {
    if (command === "join-project") {
      const data = await emit(socket, "joinProject", { project_id: payload });
      const project = data[0];
      process.stdout.write(JSON.stringify({ ok: true, project }));
    } else if (command === "join-doc") {
      const data = await emit(socket, "joinDoc", payload, { encodeRanges: true });
      const docLinesAscii = data[0] || [];
      const version = data[1];
      const content = docLinesAscii
        .map((line) => Buffer.from(line, "latin1").toString("utf8"))
        .join("\n");
      process.stdout.write(JSON.stringify({ ok: true, content, version }));
    } else {
      fail(`Unsupported command: ${command}`);
    }
  } finally {
    socket.disconnect();
  }
}

main().catch((err) => fail(err && err.stack ? err.stack : err));
