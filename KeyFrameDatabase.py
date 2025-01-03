import threading

import numpy as np

class KeyFrameDatabase:
    def __init__(self, voc):
        self.mpVoc = voc
        self.mMutex = threading.Lock()
        self.mvInvertedFile = {}  # Dictionary to store inverted file data

    def add(self, pKF):
        """Add a keyframe to the database."""
        with self.mMutex:  # Ensure thread safety
            for word_id in pKF.mBowVec.keys():
                if word_id not in self.mvInvertedFile:
                    self.mvInvertedFile[word_id] = []

                self.mvInvertedFile[word_id].append(pKF)

    def erase(self, pKF):
        """Erase a keyframe from the database."""
        with self.mMutex:  # Ensure thread safety
            # Iterate over the words in the BoW vector of the keyframe
            for word_id in pKF.mBowVec.keys():
                # Get the list of keyframes sharing the word
                if word_id in self.mvInvertedFile:
                    lKFs = self.mvInvertedFile[word_id]
                    # Remove the keyframe if it exists in the list
                    if pKF in lKFs:
                        lKFs.remove(pKF)

    def clear(self):
        """Clear the inverted file and resize it to the vocabulary size."""
        self.mvInvertedFile = {i: [] for i in range(self.vocabulary_size)}

    def detect_loop_candidates(self, pKF, min_score):
        """Detect loop candidates for a given KeyFrame."""
        spConnectedKeyFrames = pKF.get_connected_key_frames()
        lKFsSharingWords = []
        #print("-->", self.mvInvertedFile)
        # Search all keyframes that share a word with the current keyframe
        # Discard keyframes connected to the query keyframe
        with self.mMutex:
            for word_id in pKF.mBowVec.keys():
                if word_id not in self.mvInvertedFile:
                    continue

                lKFs = self.mvInvertedFile[word_id]
                for pKFi in lKFs:
                    if pKFi.mnLoopQuery != pKF.mnId:
                        pKFi.mnLoopWords = 0
                        if pKFi not in spConnectedKeyFrames: #not in for debug in
                            pKFi.mnLoopQuery = pKF.mnId
                            lKFsSharingWords.append(pKFi)
                    pKFi.mnLoopWords += 1

        print("-->", lKFsSharingWords)
        if not lKFsSharingWords:
            return []

        # Determine the maximum number of common words
        max_common_words = max(pKFi.mnLoopWords for pKFi in lKFsSharingWords)
        min_common_words = int(max_common_words * 0.8)

        # Compute similarity scores and filter matches
        lScoreAndMatch = []
        for pKFi in lKFsSharingWords:
            if pKFi.mnLoopWords > min_common_words:
                score = self.mpVoc.score(pKF.mBowVec, pKFi.mBowVec)
                pKFi.mLoopScore = score
                if score >= min_score:
                    lScoreAndMatch.append((score, pKFi))

        if not lScoreAndMatch:
            return []

        # Accumulate scores based on covisibility
        lAccScoreAndMatch = []
        best_acc_score = min_score

        for score, pKFi in lScoreAndMatch:
            vpNeighs = pKFi.get_best_covisibility_key_frames(10)

            acc_score = score
            best_score = score
            pBestKF = pKFi

            for pKF2 in vpNeighs:
                if pKF2.mnLoopQuery == pKF.mnId and pKF2.mnLoopWords > min_common_words:
                    acc_score += pKF2.mLoopScore
                    if pKF2.mLoopScore > best_score:
                        pBestKF = pKF2
                        best_score = pKF2.mLoopScore

            lAccScoreAndMatch.append((acc_score, pBestKF))
            if acc_score > best_acc_score:
                best_acc_score = acc_score

        # Filter keyframes based on accumulated score
        min_score_to_retain = 0.75 * best_acc_score
        spAlreadyAddedKF = set()
        vpLoopCandidates = []

        for acc_score, pKFi in lAccScoreAndMatch:
            if acc_score > min_score_to_retain and pKFi not in spAlreadyAddedKF:
                vpLoopCandidates.append(pKFi)
                spAlreadyAddedKF.add(pKFi)

        return vpLoopCandidates

    def detect_relocalization_candidates(self, F):
        """Detect relocalization candidates for a given frame."""
        lKFsSharingWords = []

        # Search all keyframes that share a word with the current frame
        with self.mMutex:
            for word_id in F.mBowVec.keys():
                if word_id not in self.mvInvertedFile:
                    continue

                lKFs = self.mvInvertedFile[word_id]
                for pKFi in lKFs:
                    if pKFi.mnRelocQuery != F.mnId:
                        pKFi.mnRelocWords = 0
                        pKFi.mnRelocQuery = F.mnId
                        lKFsSharingWords.append(pKFi)
                    pKFi.mnRelocWords += 1

        print("-->", lKFsSharingWords)

        if not lKFsSharingWords:
            return []

        # Determine the maximum number of common words
        maxCommonWords = max(pKFi.mnRelocWords for pKFi in lKFsSharingWords)
        minCommonWords = int(maxCommonWords * 0.8)

        # Compute similarity scores
        lScoreAndMatch = []
        for pKFi in lKFsSharingWords:
            if pKFi.mnRelocWords > minCommonWords:
                score = self.mpVoc.score(F.mBowVec, pKFi.mBowVec)
                pKFi.mRelocScore = score
                lScoreAndMatch.append((score, pKFi))

        if not lScoreAndMatch:
            return []

        # Accumulate scores based on covisibility
        lAccScoreAndMatch = []
        bestAccScore = 0

        for score, pKFi in lScoreAndMatch:
            vpNeighs = pKFi.get_best_covisibility_key_frames(10)

            accScore = score
            bestScore = score
            pBestKF = pKFi

            for pKF2 in vpNeighs:
                if pKF2.mnRelocQuery != F.mnId:
                    continue

                accScore += pKF2.mRelocScore
                if pKF2.mRelocScore > bestScore:
                    pBestKF = pKF2
                    bestScore = pKF2.mRelocScore

            lAccScoreAndMatch.append((accScore, pBestKF))
            bestAccScore = max(bestAccScore, accScore)

        # Return keyframes with score higher than 0.75 * bestAccScore
        minScoreToRetain = 0.75 * bestAccScore
        spAlreadyAddedKF = set()
        vpRelocCandidates = []

        for accScore, pKFi in lAccScoreAndMatch:
            if accScore > minScoreToRetain and pKFi not in spAlreadyAddedKF:
                vpRelocCandidates.append(pKFi)
                spAlreadyAddedKF.add(pKFi)

        return vpRelocCandidates
