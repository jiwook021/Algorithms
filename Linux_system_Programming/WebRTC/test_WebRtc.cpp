#include "WebRtc.hpp"
#include <cassert>
#include <iostream>

using namespace WebRtc;

// ============================================================================
// SDP type parsing
// ============================================================================

void TestParseSdpTypeOffer() {
    assert(ParseSdpType("offer") == SdpType::OFFER);
    std::cout << "[PASS] TestParseSdpTypeOffer" << std::endl;
}

void TestParseSdpTypeAnswer() {
    assert(ParseSdpType("answer") == SdpType::ANSWER);
    std::cout << "[PASS] TestParseSdpTypeAnswer" << std::endl;
}

void TestParseSdpTypeUnknown() {
    assert(ParseSdpType("pranswer") == SdpType::UNKNOWN);
    assert(ParseSdpType("") == SdpType::UNKNOWN);
    std::cout << "[PASS] TestParseSdpTypeUnknown" << std::endl;
}

void TestSdpTypeRoundTrip() {
    assert(SdpTypeToString(ParseSdpType("offer")) == "offer");
    assert(SdpTypeToString(ParseSdpType("answer")) == "answer");
    std::cout << "[PASS] TestSdpTypeRoundTrip" << std::endl;
}

// ============================================================================
// SDP validation
// ============================================================================

void TestValidateSdpBlobValid() {
    std::string sdp = "v=0\r\no=- 123 456 IN IP4 127.0.0.1\r\n";
    assert(ValidateSdpBlob(sdp) == true);
    std::cout << "[PASS] TestValidateSdpBlobValid" << std::endl;
}

void TestValidateSdpBlobEmpty() {
    assert(ValidateSdpBlob("") == false);
    std::cout << "[PASS] TestValidateSdpBlobEmpty" << std::endl;
}

void TestValidateSdpBlobNoVersion() {
    assert(ValidateSdpBlob("o=- 123 IN IP4 127.0.0.1") == false);
    std::cout << "[PASS] TestValidateSdpBlobNoVersion" << std::endl;
}

void TestValidateSdpBlobLeadingWhitespace() {
    assert(ValidateSdpBlob("  \n v=0\r\n") == true);
    std::cout << "[PASS] TestValidateSdpBlobLeadingWhitespace" << std::endl;
}

// ============================================================================
// ICE candidate parsing
// ============================================================================

void TestParseIceCandidateLineValid() {
    auto result = ParseIceCandidateLine("0 candidate:123 1 udp 456 192.168.1.1 5000 typ host");
    assert(result.has_value());
    assert(result->sdpMlineIndex == 0);
    assert(result->candidate.find("candidate:123") != std::string::npos);
    std::cout << "[PASS] TestParseIceCandidateLineValid" << std::endl;
}

void TestParseIceCandidateLineNoCandidate() {
    auto result = ParseIceCandidateLine("0");
    assert(!result.has_value());
    std::cout << "[PASS] TestParseIceCandidateLineNoCandidate" << std::endl;
}

void TestParseIceCandidateLineEmpty() {
    auto result = ParseIceCandidateLine("");
    assert(!result.has_value());
    std::cout << "[PASS] TestParseIceCandidateLineEmpty" << std::endl;
}

void TestParseIceCandidateLineNonNumeric() {
    auto result = ParseIceCandidateLine("abc candidate:xyz");
    assert(!result.has_value());
    std::cout << "[PASS] TestParseIceCandidateLineNonNumeric" << std::endl;
}

void TestValidateIceCandidateStringValid() {
    assert(ValidateIceCandidateString("candidate:123 1 udp 456 192.168.1.1 5000 typ host") == true);
    std::cout << "[PASS] TestValidateIceCandidateStringValid" << std::endl;
}

void TestValidateIceCandidateStringWithPrefix() {
    assert(ValidateIceCandidateString("a=candidate:123 1 udp 456 10.0.0.1 9000 typ srflx") == true);
    std::cout << "[PASS] TestValidateIceCandidateStringWithPrefix" << std::endl;
}

void TestValidateIceCandidateStringInvalid() {
    assert(ValidateIceCandidateString("not-a-candidate") == false);
    assert(ValidateIceCandidateString("") == false);
    std::cout << "[PASS] TestValidateIceCandidateStringInvalid" << std::endl;
}

// ============================================================================
// CLI command parsing
// ============================================================================

void TestParseCliCommandStart() {
    assert(ParseCliCommand("start") == CliCommand::START);
    assert(ParseCliCommand("  START  ") == CliCommand::START);
    std::cout << "[PASS] TestParseCliCommandStart" << std::endl;
}

void TestParseCliCommandCreateOffer() {
    assert(ParseCliCommand("create-offer") == CliCommand::CREATE_OFFER);
    std::cout << "[PASS] TestParseCliCommandCreateOffer" << std::endl;
}

void TestParseCliCommandStop() {
    assert(ParseCliCommand("stop") == CliCommand::STOP);
    std::cout << "[PASS] TestParseCliCommandStop" << std::endl;
}

void TestParseCliCommandOffer() {
    assert(ParseCliCommand("offer") == CliCommand::SET_OFFER);
    std::cout << "[PASS] TestParseCliCommandOffer" << std::endl;
}

void TestParseCliCommandAnswer() {
    assert(ParseCliCommand("answer") == CliCommand::SET_ANSWER);
    std::cout << "[PASS] TestParseCliCommandAnswer" << std::endl;
}

void TestParseCliCommandIce() {
    assert(ParseCliCommand("ice 0 candidate:abc") == CliCommand::ADD_ICE);
    std::cout << "[PASS] TestParseCliCommandIce" << std::endl;
}

void TestParseCliCommandQuit() {
    assert(ParseCliCommand("quit") == CliCommand::QUIT);
    std::cout << "[PASS] TestParseCliCommandQuit" << std::endl;
}

void TestParseCliCommandUnknown() {
    assert(ParseCliCommand("foobar") == CliCommand::UNKNOWN);
    assert(ParseCliCommand("") == CliCommand::UNKNOWN);
    std::cout << "[PASS] TestParseCliCommandUnknown" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== WebRtc Unit Tests ===" << std::endl;

    TestParseSdpTypeOffer();
    TestParseSdpTypeAnswer();
    TestParseSdpTypeUnknown();
    TestSdpTypeRoundTrip();

    TestValidateSdpBlobValid();
    TestValidateSdpBlobEmpty();
    TestValidateSdpBlobNoVersion();
    TestValidateSdpBlobLeadingWhitespace();

    TestParseIceCandidateLineValid();
    TestParseIceCandidateLineNoCandidate();
    TestParseIceCandidateLineEmpty();
    TestParseIceCandidateLineNonNumeric();
    TestValidateIceCandidateStringValid();
    TestValidateIceCandidateStringWithPrefix();
    TestValidateIceCandidateStringInvalid();

    TestParseCliCommandStart();
    TestParseCliCommandCreateOffer();
    TestParseCliCommandStop();
    TestParseCliCommandOffer();
    TestParseCliCommandAnswer();
    TestParseCliCommandIce();
    TestParseCliCommandQuit();
    TestParseCliCommandUnknown();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
